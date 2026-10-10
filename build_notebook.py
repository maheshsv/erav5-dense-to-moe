"""Builds dense_to_moe.ipynb for ERA Session 14 (train dense, upcycle to MoE)."""
import nbformat as nbf

REPO = "https://github.com/maheshsv/erav5-dense-to-moe.git"
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# From a Linear Model to a Mixture of Experts — ERA Session 14

Train a dense byte-level transformer (its FFNs are plain `Linear → GELU → Linear` — the
"linear model"), then **upcycle** it into a Mixture-of-Experts: every FFN becomes E=4 expert
copies of itself behind a top-2 router. Keep training. Prove two things:

1. the conversion is **exactly loss-preserving** (no cliff at the switch), and
2. the MoE **keeps training and keeps reducing loss** — measured against the honest control,
   the same dense checkpoint continued on the same extra tokens.

Three resumable runs, each logged to `runs/*/metrics.jsonl`:

| run | phase | tokens | starts from |
|---|---|---|---|
| 1 | dense (Linear FFNs), from scratch | 20M | — |
| 2 | dense-continue (control) | +20M | run 1 final weights |
| 3 | **MoE** (E=4, top-2, upcycled) | +20M | run 1 final weights |

On a machine with the committed logs this notebook loads and plots; on a fresh Colab it
clones the repo and trains all three for real (`train_run` resumes from checkpoints and
skips itself once a run's `DONE` flag exists).""")

code(f"""import json, os, subprocess
if not os.path.exists("moe.py"):                          # Colab: fetch the code
    subprocess.run(["git", "clone", "{REPO}"], check=True)
    os.chdir("erav5-dense-to-moe")
import torch
import moe
print("device:", "cuda" if torch.cuda.is_available() else
      "mps" if torch.backends.mps.is_available() else "cpu", "| torch", torch.__version__)""")

md("""## The conversion, shown on a tiny model first

`LM.upcycle()` replaces each `FFN` with an `MoE` whose experts are `deepcopy`s of that FFN
and whose router is a fresh `Linear(d, E)`. Top-k weights are renormalized to sum to 1, so
with identical experts the output is **the dense output** — whatever the router says.""")

code("""torch.manual_seed(0)
tiny = moe.LM(d=64, L=2, heads=4, ctx=32)
x = torch.randint(0, 256, (4, 32))
with torch.no_grad():
    before = tiny(x).clone()
tiny.upcycle(n_experts=4, top_k=2)
with torch.no_grad():
    after = tiny(x)
print(f"max |dense output - MoE output| right after conversion: {(before - after).abs().max():.2e}")
tot, act = tiny.n_params()
print(f"tiny model params: {tot:,} total, {act:,} active per token")""")

md("""## Run 1 — the dense ("linear") model, 20M tokens""")
code("""moe.train_run("runs/1_dense", "dense", total_tokens=int(20e6), lr=1e-3, warmup=200)""")

md("""## Run 2 — control: keep training the dense model (+20M tokens)""")
code("""moe.train_run("runs/2_dense_continue", "dense-continue", total_tokens=int(20e6), lr=5e-4,
              warmup=100, init_from="runs/1_dense/model_final.pt")""")

md("""## Run 3 — convert to MoE, keep training (+20M tokens)

The run prints the conversion check on the real model — dense val loss vs MoE val loss at
step 0 — before a single MoE step is taken.""")
code("""moe.train_run("runs/3_moe", "moe", total_tokens=int(20e6), lr=5e-4, warmup=100,
              init_from="runs/1_dense/model_final.pt", n_experts=4, top_k=2)""")

md("""## Results""")
code("""import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load(out):
    by_step = {}
    for line in open(os.path.join(out, "metrics.jsonl")):
        r = json.loads(line); by_step[r["step"]] = r
    return [by_step[s] for s in sorted(by_step)], json.load(open(os.path.join(out, "DONE")))

r1, f1 = load("runs/1_dense")
r2, f2 = load("runs/2_dense_continue")
r3, f3 = load("runs/3_moe")
T1 = f1["tokens"]

print("conversion check on the trained model:", json.dumps(f3["conversion"], indent=2))
print()
print(f"{'run':<22}{'params total':>16}{'active':>13}{'final val':>11}{'tok/s':>9}{'peak mem':>10}")
for name, f, recs in (("1 dense", f1, r1), ("2 dense-continue", f2, r2), ("3 MoE E=4 top-2", f3, r3)):
    tps = sorted(r["tok_per_s"] for r in recs if r["tok_per_s"] > 0)[len(recs) // 2]
    print(f"{name:<22}{f['params_total']:>16,}{f['params_active']:>13,}{f['final_val_loss']:>11.4f}"
          f"{tps:>9,}{f['peak_mem_gb']:>9.2f}G")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
ax = axes[0]
ax.plot([r["tokens"] / 1e6 for r in r1], [r["val_loss"] for r in r1], color="tab:blue", lw=2,
        label="1 dense (Linear FFN)")
ax.plot([(T1 + r["tokens"]) / 1e6 for r in r2], [r["val_loss"] for r in r2], color="tab:gray",
        lw=2, label="2 dense, continued (control)")
ax.plot([(T1 + r["tokens"]) / 1e6 for r in r3], [r["val_loss"] for r in r3], color="tab:red",
        lw=2, label="3 MoE, upcycled from run 1")
ax.axvline(T1 / 1e6, color="k", ls="--", lw=1)
ax.annotate("convert to MoE", (T1 / 1e6, max(r["val_loss"] for r in r2)),
            textcoords="offset points", xytext=(6, 0))
ax.set(xlabel="total tokens seen (M)", ylabel="val loss",
       title="Loss keeps falling through the conversion - faster as an MoE")
ax.legend(); ax.grid(alpha=0.3)
ax.set_ylim(min(r["val_loss"] for r in r3) - 0.05, 2.2)

ax = axes[1]
usage = [r["usage"] for r in r3]                    # [log][layer][expert]
steps = [r["step"] for r in r3]
L = len(usage[0]); E = len(usage[0][0])
for e in range(E):
    ax.plot(steps, [sum(u[l][e] for l in range(L)) / L for u in usage], label=f"expert {e}")
ax.axhline(2 / E, color="gray", ls="--", lw=1, label="perfect balance (top-2 of 4)")
ax.set(xlabel="MoE step", ylabel="fraction of tokens routed (avg over layers)",
       title="Router load: experts specialize but stay used")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("dense_to_moe.png", dpi=130); plt.show()

v_conv, v_dense_end, v_moe_end = f3["conversion"]["moe_val_after"], f2["final_val_loss"], f3["final_val_loss"]
summary = {"dense_20M_val": f1["final_val_loss"], "conversion": f3["conversion"],
           "dense_continue_40M_val": v_dense_end, "moe_40M_val": v_moe_end,
           "moe_gain_over_dense_control": round(v_dense_end - v_moe_end, 4),
           "moe_drop_after_conversion": round(v_conv - v_moe_end, 4)}
json.dump(summary, open("summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
print("saved dense_to_moe.png, summary.json")""")

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}, "colab": {"provenance": []}})
nbf.write(nb, "dense_to_moe.ipynb")
print("wrote dense_to_moe.ipynb")
