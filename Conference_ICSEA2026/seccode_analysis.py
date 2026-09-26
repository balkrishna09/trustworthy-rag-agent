"""
Analyze the secure-coding SDLC use-case runs and emit the LaTeX results table
plus real poisoned-document examples for the paper.

Metrics are recomputed from per-sample results (is_poisoned_set vs is_trustworthy)
using the same method as the main paper. Run after run_seccode_usecase.py:
  venv/Scripts/python.exe Conference_ICSEA2026/seccode_analysis.py
"""
import sys, os, json, glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
OUT = "Conference_ICSEA2026"


def metrics(truth, flagged):
    tp = sum(1 for t, f in zip(truth, flagged) if t and f)
    fp = sum(1 for t, f in zip(truth, flagged) if (not t) and f)
    fn = sum(1 for t, f in zip(truth, flagged) if t and (not f))
    tn = sum(1 for t, f in zip(truth, flagged) if (not t) and (not f))
    n = tp + fp + fn + tn
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * prec * rec / (prec + rec)) if tp > 0 else 0.0
    return dict(acc=acc, prec=prec, rec=rec, f1=f1, tp=tp, fp=fp, fn=fn, tn=tn)


def load_latest():
    """Latest seccode result per strategy (skip the small smoke runs)."""
    best = {}
    for f in glob.glob("data/experiments/experiment_seccode_*.json"):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        sr = d.get("sample_results", [])
        if len(sr) < 30:   # skip 4-doc smoke runs
            continue
        strat = d.get("config", {}).get("name", "").replace("seccode_", "")
        ts = d.get("timestamp", os.path.getmtime(f))
        if strat not in best or ts > best[strat][0]:
            best[strat] = (ts, d)
    return {s: d for s, (ts, d) in best.items()}


def pc(x):
    return "n/a" if (x is None or (isinstance(x, float) and x != x)) else f"{100*x:.1f}"


runs = load_latest()
order = ["injection", "contradiction", "subtle", "entity_swap", "mixed"]
label = {"injection": "Instruction injection", "contradiction": "Contradiction",
         "subtle": "Subtle manipulation", "entity_swap": "Entity swap", "mixed": "Mixed"}

print(f"{'strategy':14} {'acc':>6} {'prec':>6} {'rec':>6} {'f1':>6} {'sep':>6}  (TP/FP/FN/TN)")
rows = []
for s in order:
    if s not in runs:
        print(f"{s:14}  (no full run found)")
        continue
    sr = runs[s]["sample_results"]
    truth = [bool(x["is_poisoned_set"]) for x in sr]
    flagged = [not bool(x.get("is_trustworthy", x["trust_score"] >= 0.5)) for x in sr]
    m = metrics(truth, flagged)
    clean = [x["trust_score"] for x in sr if not x["is_poisoned_set"]]
    pois = [x["trust_score"] for x in sr if x["is_poisoned_set"]]
    sep = (sum(clean) / len(clean) - sum(pois) / len(pois)) if clean and pois else float("nan")
    rows.append((s, m, sep, len(sr)))
    print(f"{s:14} {pc(m['acc']):>6} {pc(m['prec']):>6} {pc(m['rec']):>6} {pc(m['f1']):>6} {sep:>6.3f}  "
          f"({m['tp']}/{m['fp']}/{m['fn']}/{m['tn']})  N={len(sr)}")

# ---- emit LaTeX table ----
lines = ["% Auto-generated: secure-coding SDLC use-case results"]
lines.append("\\begin{table}[t]\\caption{Secure-coding RAG assistant: poison detection by attack strategy "
             "(40 OWASP/CWE secure-coding rules; Llama~3.3~70B + MiniLM, $K{=}5$).}\\label{tab:seccode}\\centering")
lines.append("\\resizebox{\\columnwidth}{!}{%")
lines.append("\\begin{tabular}{lccccc}\\toprule")
lines.append("Strategy & Acc. & Prec. & Rec. & F1 & $\\Delta$ \\\\\\midrule")
for s, m, sep, _ in rows:
    seps = "n/a" if sep != sep else f"{sep:.3f}"
    lines.append(f"{label[s]} & {pc(m['acc'])} & {pc(m['prec'])} & {pc(m['rec'])} & {pc(m['f1'])} & {seps} \\\\")
lines.append("\\bottomrule\\end{tabular}}\\end{table}")
open(os.path.join(OUT, "seccode_results.tex"), "w", encoding="utf-8").write("\n".join(lines))
print("\nWrote", os.path.join(OUT, "seccode_results.tex"))

# ---- real poisoned-document examples (no LLM needed) ----
print("\n=== Example poisoned secure-coding documents ===")
try:
    from src.experiments.poisoned_dataset import PoisonedDatasetGenerator, PoisonStrategy
    kb = []
    for ln in open("data/raw/secure_coding/secure_coding.jsonl", encoding="utf-8"):
        o = json.loads(ln); kb.append(f"{o['question']} {o['best_answer']}")
    for strat in [PoisonStrategy.INSTRUCTION_INJECTION, PoisonStrategy.CONTRADICTION,
                  PoisonStrategy.ENTITY_SWAP, PoisonStrategy.SUBTLE_MANIPULATION]:
        gen = PoisonedDatasetGenerator()
        samples, _ = gen.create_poisoned_dataset(kb, poison_ratio=0.3, strategy=strat)
        for i, sm in enumerate(samples):
            pt = getattr(sm, "poisoned_text", None)
            if pt and pt != kb[i]:
                print(f"\n[{strat.value}]")
                print(f"  ORIG: {kb[i][:140]}")
                print(f"  POIS: {pt[:200]}")
                break
except Exception as e:
    print("  (could not generate examples:", e, ")")
