"""moe — train a dense byte-level transformer, then upcycle its FFNs into a
Mixture-of-Experts and keep training.

Phases (each a resumable run with JSONL logs):
  dense          : standard transformer (Linear FFNs), from scratch.
  dense-continue : control — keep training the dense checkpoint.
  moe            : convert the dense checkpoint into an MoE (each FFN -> E expert
                   copies + a top-k router), then keep training.

Conversion is exactly loss-preserving: experts start as copies of the trained FFN
and the top-k router weights are renormalized to sum to 1, so at step 0 the MoE
output == the dense output. The experts then diverge because the router sends
them different tokens. fp32 everywhere; mps / cuda / cpu.
"""
import copy
import json
import math
import os
import time
import urllib.request
import zipfile

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------- data
ENWIK8_URL = "http://mattmahoney.net/dc/enwik8.zip"

def load_enwik8(data_dir="data"):
    raw = os.path.join(data_dir, "enwik8")
    if not os.path.exists(raw):
        os.makedirs(data_dir, exist_ok=True)
        zpath = raw + ".zip"
        print("downloading enwik8 (36 MB)...")
        urllib.request.urlretrieve(ENWIK8_URL, zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extract("enwik8", data_dir)
        os.remove(zpath)
    data = torch.frombuffer(bytearray(open(raw, "rb").read()), dtype=torch.uint8).long()
    return data[:95_000_000], data[95_000_000:]

def batcher(data, batch, ctx, generator):
    starts = torch.randint(0, len(data) - ctx - 1, (batch,), generator=generator)
    x = torch.stack([data[s:s + ctx + 1] for s in starts])
    return x[:, :-1], x[:, 1:]

# ---------------------------------------------------------------- modules
class Attn(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.qkv, self.proj, self.h = nn.Linear(d, 3 * d), nn.Linear(d, d), heads

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.h, C // self.h).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(a.transpose(1, 2).reshape(B, T, C))


class FFN(nn.Module):
    """The 'linear model' part: two Linear layers with a GELU between."""
    def __init__(self, d):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class MoE(nn.Module):
    """Top-k mixture of FFN experts. Built FROM a trained FFN (sparse upcycling)."""
    def __init__(self, dense_ffn, d, n_experts=4, top_k=2):
        super().__init__()
        self.E, self.k = n_experts, top_k
        self.experts = nn.ModuleList(copy.deepcopy(dense_ffn) for _ in range(n_experts))
        self.router = nn.Linear(d, n_experts, bias=False)
        nn.init.normal_(self.router.weight, std=0.02)
        self.aux = torch.zeros(())
        self.usage = torch.zeros(n_experts)

    def forward(self, h):
        B, T, C = h.shape
        x = h.reshape(-1, C)
        probs = F.softmax(self.router(x), dim=-1)                # [N, E]
        topv, topi = probs.topk(self.k, dim=-1)                  # [N, k]
        topv = topv / topv.sum(-1, keepdim=True)                 # weights sum to 1
        # Dense-masked evaluation: every expert sees every token, weighted by its
        # (renormalized) top-k gate, which is exactly 0 for non-selected experts, so
        # outputs AND gradients equal sparse routing. Static shapes on purpose: the
        # gather/index_add version bloats the MPS caching allocator (new sub-batch
        # shape every step) until the machine swaps.
        gate = torch.zeros_like(probs).scatter(1, topi, topv)   # [N, E], rows sum to 1
        y = torch.zeros_like(x)
        for e in range(self.E):
            y = y + gate[:, e:e + 1] * self.experts[e](x)
        # Switch-style load-balancing loss: E * sum_e f_e * P_e  (== top_k when balanced)
        f = (gate > 0).float().mean(0)                           # fraction routed to e
        P = probs.mean(0)
        self.aux = self.E * (f * P).sum()
        self.usage = f.detach().cpu()
        return y.view(B, T, C)


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attn(d, heads)
        self.ffn = FFN(d)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class LM(nn.Module):
    def __init__(self, V=256, d=512, L=6, heads=8, ctx=256):
        super().__init__()
        self.d = d
        self.emb, self.pos = nn.Embedding(V, d), nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(L))
        self.ln_f, self.head = nn.LayerNorm(d), nn.Linear(d, V, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.emb(x) + self.pos(torch.arange(x.shape[1], device=x.device))
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.ln_f(h))

    def is_moe(self):
        return isinstance(self.blocks[0].ffn, MoE)

    def aux_loss(self):
        return sum(b.ffn.aux for b in self.blocks) / len(self.blocks)

    def expert_usage(self):
        return [b.ffn.usage.tolist() for b in self.blocks]

    def upcycle(self, n_experts=4, top_k=2):
        """Replace every FFN by an MoE whose experts are copies of it. In place."""
        for blk in self.blocks:
            blk.ffn = MoE(blk.ffn, self.d, n_experts, top_k)
        return self

    def n_params(self):
        total = sum(p.numel() for p in self.parameters())
        if not self.is_moe():
            return total, total
        per_expert = sum(p.numel() for p in self.blocks[0].ffn.experts[0].parameters())
        E, k = self.blocks[0].ffn.E, self.blocks[0].ffn.k
        active = total - len(self.blocks) * (E - k) * per_expert
        return total, active

# ---------------------------------------------------------------- training
def mem_bytes(device):
    if device == "mps":
        return torch.mps.driver_allocated_memory()
    if device == "cuda":
        return torch.cuda.max_memory_allocated()
    return 0

@torch.no_grad()
def eval_loss(model, xv, yv):
    return F.cross_entropy(model(xv).reshape(-1, 256), yv.reshape(-1)).item()

def train_run(out_dir, phase, total_tokens, lr, batch=64, ctx=256, warmup=200,
              init_from=None, n_experts=4, top_k=2, aux_coef=0.01, device=None,
              data_dir="data", log_every=20, ckpt_every=400, seed=1234):
    device = device or ("cuda" if torch.cuda.is_available()
                        else "mps" if torch.backends.mps.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)
    done_flag = os.path.join(out_dir, "DONE")
    metrics_path = os.path.join(out_dir, "metrics.jsonl")
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    if os.path.exists(done_flag):
        print(f"[{out_dir}] already complete - skipping")
        return metrics_path

    train_data, val_data = load_enwik8(data_dir)
    xv, yv = batcher(val_data, 64, ctx, torch.Generator().manual_seed(999))
    xv, yv = xv.to(device), yv.to(device)

    torch.manual_seed(seed)
    model = LM(ctx=ctx).to(device)
    conversion = None
    if init_from is not None:
        model.load_state_dict(torch.load(init_from, map_location=device, weights_only=True))
        dense_val = eval_loss(model, xv, yv)
        if phase == "moe":
            model.upcycle(n_experts, top_k).to(device)
            moe_val = eval_loss(model, xv, yv)
            conversion = {"dense_val_before": dense_val, "moe_val_after": moe_val,
                          "abs_diff": abs(dense_val - moe_val)}
            print(f"[conversion] dense val {dense_val:.6f} -> MoE val {moe_val:.6f} "
                  f"(|diff| {abs(dense_val - moe_val):.2e})")
        else:
            conversion = {"dense_val_before": dense_val}
    total, active = model.n_params()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    steps_total = math.ceil(total_tokens / (batch * ctx))
    sched = lambda s: (s / warmup if s < warmup else
                       0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warmup) /
                                                       max(1, steps_total - warmup))))
    start_step = 0
    g = torch.Generator().manual_seed(seed + (7 if init_from else 0))
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start_step = ck["step"]; g.set_state(ck["rng"].cpu().to(torch.uint8))
        print(f"[{out_dir}] resuming from step {start_step}")

    peak, t_run = 0, time.time()
    tok_done = start_step * batch * ctx
    logf = open(metrics_path, "a")
    win_t0, win_tok = time.time(), 0
    for step in range(start_step + 1, steps_total + 1):
        for gp in opt.param_groups:
            gp["lr"] = lr * sched(step)
        x, y = batcher(train_data, batch, ctx, g)
        x, y = x.to(device), y.to(device)
        logits = model(x)
        ce = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        aux = model.aux_loss() if model.is_moe() else torch.zeros((), device=device)
        loss = ce + aux_coef * aux
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tok_done += batch * ctx; win_tok += batch * ctx
        peak = max(peak, mem_bytes(device))
        if device == "mps" and step % 50 == 0:
            torch.mps.empty_cache()                    # keep the caching allocator honest
        if step % log_every == 0 or step == steps_total:
            if device == "mps":
                torch.mps.synchronize()
            dt = time.time() - win_t0
            tps = win_tok / dt if dt > 0 else 0
            win_t0, win_tok = time.time(), 0
            vloss = eval_loss(model, xv, yv)
            rec = {"step": step, "tokens": tok_done, "loss": round(ce.item(), 4),
                   "val_loss": round(vloss, 4), "tok_per_s": round(tps),
                   "peak_mem_gb": round(peak / 2**30, 2), "lr": round(lr * sched(step), 6)}
            if model.is_moe():
                rec["aux"] = round(aux.item(), 4)
                rec["usage"] = [[round(u, 3) for u in layer] for layer in model.expert_usage()]
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            if step % (log_every * 10) == 0 or step == steps_total:
                print(f"[{phase}] step {step}/{steps_total} tok {tok_done/1e6:.1f}M "
                      f"loss {ce.item():.4f} val {vloss:.4f} {tps:,.0f} tok/s "
                      f"peak {peak/2**30:.2f} GB", flush=True)
        if step % ckpt_every == 0:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": step, "rng": g.get_state()}, ckpt_path)
    logf.close()
    torch.save(model.state_dict(), os.path.join(out_dir, "model_final.pt"))
    final = {"phase": phase, "batch": batch, "params_total": total, "params_active": active,
             "steps": steps_total, "tokens": tok_done, "final_train_loss": round(ce.item(), 4),
             "final_val_loss": round(vloss, 4), "peak_mem_gb": round(peak / 2**30, 2),
             "wall_clock_s": round(time.time() - t_run), "device": device, "lr": lr,
             "conversion": conversion}
    with open(done_flag, "w") as fdone:
        json.dump(final, fdone, indent=2)
    print(f"[{out_dir}] DONE: {final}", flush=True)
    return metrics_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["dense", "dense-continue", "moe"])
    ap.add_argument("--tokens", type=float, default=20e6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--init", default=None)
    ap.add_argument("--experts", type=int, default=4)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    train_run(a.out, a.phase, int(a.tokens), a.lr, warmup=a.warmup, init_from=a.init,
              n_experts=a.experts, top_k=a.topk)
