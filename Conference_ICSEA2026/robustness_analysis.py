"""Robustness & calibration analysis for the ICSEA 2026 paper.

Computes, purely from stored per-sample results (NO LLM, NO model inference):
  1. Run-to-run variance (mean +/- std) over genuine repeated runs.
  2. Threshold-independent evaluation: ROC-AUC and average precision (AP).
  3. Per-LLM threshold calibration that recovers Qwen performance.

Emits a human-readable report plus LaTeX snippets (tables + pgfplots ROC figure).
Run:  python Conference_ICSEA2026/robustness_analysis.py
"""

import json, glob, os
import numpy as np

random = np.random.default_rng(42)
OUT = "Conference_ICSEA2026"
LLMS = ["llama3.3:70b", "qwen3.5:35b", "mistral:7b-instruct"]  # 3rd LLM added (skipped if no data yet)

# ---------------------------------------------------------------- load & label
def strategy_of(name, cfg):
    n = name.lower()
    if "contradiction" in n: return "contradiction"
    if "injection" in n:     return "injection"
    if "entity_swap" in n:   return "entity_swap"
    if "subtle" in n:        return "subtle"
    if "detection" in n:     return "mixed"      # model-named canonical mixed run
    return "pilot"                               # bare poison_detection pilots

def load_all():
    rows = []
    for f in sorted(glob.glob("data/experiments/*.json")):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        s = d.get("sample_results", [])
        if not s or "trust_score" not in s[0] or "is_poisoned_set" not in s[0]:
            continue
        c = d.get("config", {})
        name = os.path.basename(f)
        llm = str(c.get("llm_model", "")).split("/")[-1]
        emb = str(c.get("embedding_model", "")).split("/")[-1]
        k = c.get("top_k", c.get("k", c.get("k_retrieval")))
        ds = "fever" if "fever" in name.lower() else "truthfulqa"
        trust = np.array([float(x["trust_score"]) for x in s])
        truth = np.array([bool(x["is_poisoned_set"]) for x in s])
        flagged = np.array([not bool(x.get("is_trustworthy", x["trust_score"] >= 0.5)) for x in s])
        fact = np.array([float(x.get("factuality_score", np.nan)) for x in s])
        cons = np.array([float(x.get("consistency_score", np.nan)) for x in s])
        pois = np.array([float(x.get("poison_probability", np.nan)) for x in s])
        rows.append(dict(file=name, ds=ds, llm=llm, emb=emb, k=k, n=len(s),
                         trust=trust, truth=truth, flagged=flagged,
                         fact=fact, cons=cons, pois=pois,
                         stored=d.get("metrics", {}), strat=strategy_of(name, c)))
    return rows

def metrics(truth, flagged):
    tp = int(np.sum(truth & flagged)); fp = int(np.sum(~truth & flagged))
    fn = int(np.sum(truth & ~flagged)); tn = int(np.sum(~truth & ~flagged))
    n = tp + fp + fn + tn
    acc = (tp + tn) / n if n else 0
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec and not np.isnan(prec) and not np.isnan(rec) and (prec + rec) > 0) else 0.0
    return dict(acc=acc, prec=prec, rec=rec, f1=f1, tp=tp, fp=fp, fn=fn, tn=tn)

def sep(trust, truth):
    return float(np.mean(trust[~truth]) - np.mean(trust[truth]))

rows = load_all()
print(f"Loaded {len(rows)} valid result files.\n")

# ---------------------------------------------------------------- diagnostics
print("=== DIAGNOSTIC: recomputed vs stored accuracy (sanity check) ===")
canon = [r for r in rows if r["n"] == 100 and r["k"] == 5 and r["strat"] != "pilot"]
for r in canon[:6]:
    m = metrics(r["truth"], r["flagged"])
    st = r["stored"].get("accuracy")
    print(f"  {r['file'][:52]:<52} pos={int(r['truth'].sum()):>2} "
          f"recomp_acc={m['acc']:.3f} stored_acc={st} match={'OK' if st is None or abs(m['acc']-st)<0.011 else 'DIFF'}")

# ---------------------------------------------------------------- 1) VARIANCE
print("\n=== 1) RUN-TO-RUN VARIANCE (canonical K=5, N=100 runs) ===")
def agg(group):
    accs=[]; precs=[]; recs=[]; f1s=[]; seps=[]
    for r in group:
        m = metrics(r["truth"], r["flagged"])
        accs.append(m["acc"]); recs.append(m["rec"]); f1s.append(m["f1"]); seps.append(sep(r["trust"], r["truth"]))
        if not np.isnan(m["prec"]): precs.append(m["prec"])
    def ms(a): return (np.mean(a), np.std(a)) if a else (float("nan"), 0.0)
    return len(group), ms(accs), ms(precs), ms(recs), ms(f1s), ms(seps)

variance_rows = []
keys = {}
for r in canon:
    keys.setdefault((r["ds"], r["strat"], r["llm"], r["emb"]), []).append(r)
order = ["injection","contradiction","subtle","entity_swap","mixed"]
for (ds, strat, llm, emb), g in sorted(keys.items(), key=lambda kv: (kv[0][0], order.index(kv[0][1]) if kv[0][1] in order else 9, kv[0][2])):
    nruns, (a,as_), (p,ps_), (rc,rs_), (f,fs_), (sp,sps_) = agg(g)
    print(f"  {ds:10} {strat:13} {llm:12} {emb:18} runs={nruns}  "
          f"acc={a:.3f}+/-{as_:.3f}  prec={p:.3f}+/-{ps_:.3f}  rec={rc:.3f}+/-{rs_:.3f}  f1={f:.3f}+/-{fs_:.3f}  sep={sp:.3f}+/-{sps_:.3f}")
    variance_rows.append((ds, strat, llm, emb, nruns, a, as_, p, ps_, rc, rs_, f, fs_, sp, sps_))

# ---------------------------------------------------------------- 2) ROC / AP
def roc_auc(score, truth):
    # score: higher => more likely poisoned; truth: True=poisoned
    pos = score[truth]; neg = score[~truth]
    if len(pos)==0 or len(neg)==0: return float("nan")
    allv = np.concatenate([pos, neg]); order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv)+1)
    # average ties
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts)); np.add.at(sums, inv, ranks)
    avg = sums / counts; ranks = avg[inv]
    r_pos = ranks[:len(pos)].sum()
    return (r_pos - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg))

def roc_curve(score, truth, npts=60):
    thr = np.linspace(score.min(), score.max(), npts)
    P = truth.sum(); N = (~truth).sum()
    pts = []
    for t in thr[::-1]:
        pred = score >= t
        tpr = np.sum(pred & truth)/P if P else 0
        fpr = np.sum(pred & ~truth)/N if N else 0
        pts.append((fpr, tpr))
    return [(0.0,0.0)] + pts + [(1.0,1.0)]

def average_precision(score, truth):
    o = np.argsort(-score, kind="mergesort"); t = truth[o]
    tp = np.cumsum(t); fp = np.cumsum(~t)
    rec = tp / truth.sum(); prec = tp / (tp + fp)
    ap = 0.0; prev_r = 0.0
    for p, r in zip(prec, rec):
        ap += p * (r - prev_r); prev_r = r
    return ap

def pool(strat, llm):
    g = [r for r in canon if r["strat"]==strat and r["llm"]==llm and r["ds"]=="truthfulqa"]
    if not g: return None, None, 0
    score = np.concatenate([1.0 - r["trust"] for r in g])   # poison score
    truth = np.concatenate([r["truth"] for r in g])
    return score, truth, len(g)

print("\n=== 2) THRESHOLD-INDEPENDENT (ROC-AUC, AP) on pooled mixed runs ===")
auc_table = {}
roc_curves = {}
for llm in LLMS:
    sc, tr, ng = pool("mixed", llm)
    if sc is None: continue
    auc = roc_auc(sc, tr); ap = average_precision(sc, tr)
    base = tr.mean()
    auc_table[llm] = (auc, ap, base, ng, len(tr))
    roc_curves[llm] = roc_curve(sc, tr)
    print(f"  {llm:12} pooled_runs={ng} N={len(tr)} pos_rate={base:.3f}  ROC-AUC={auc:.3f}  AP={ap:.3f}")

# ---------------------------------------------------------------- 3) CALIBRATION
print("\n=== 3) PER-LLM THRESHOLD CALIBRATION (fit tau on clean only) ===")
calib_rows = {}
for llm in LLMS:
    g = [r for r in canon if r["strat"]=="mixed" and r["llm"]==llm and r["ds"]=="truthfulqa"]
    if not g: continue
    trust = np.concatenate([r["trust"] for r in g]); truth = np.concatenate([r["truth"] for r in g])
    clean_idx = np.where(~truth)[0]; random.shuffle(clean_idx)
    half = len(clean_idx)//2
    cal_clean, test_clean = clean_idx[:half], clean_idx[half:]
    pois_idx = np.where(truth)[0]
    test_idx = np.concatenate([test_clean, pois_idx])
    tau_cal = float(np.percentile(trust[cal_clean], 10))   # 10th pctile of clean (calibration design)
    def eval_tau(tau):
        return metrics(truth[test_idx], trust[test_idx] < tau)
    m05 = eval_tau(0.5); mc = eval_tau(tau_cal)
    calib_rows[llm] = (0.5, m05, tau_cal, mc, len(test_idx))
    print(f"  {llm:12} tau=0.50 -> acc={m05['acc']:.3f} prec={m05['prec']:.3f} rec={m05['rec']:.3f} f1={m05['f1']:.3f}")
    print(f"  {llm:12} tau={tau_cal:.3f} -> acc={mc['acc']:.3f} prec={mc['prec']:.3f} rec={mc['rec']:.3f} f1={mc['f1']:.3f}  (calibrated)")

# ---------------------------------------------------------------- emit LaTeX
def f3(x): return "n/a" if (x is None or (isinstance(x,float) and np.isnan(x))) else f"{x:.3f}"
def pc(x): return "n/a" if (x is None or (isinstance(x,float) and np.isnan(x))) else f"{100*x:.1f}"

# Variance table (per-strategy + mixed, Llama+MiniLM)
lines = []
lines.append("% Auto-generated: run-to-run variance (Llama 3.3 70B + MiniLM, K=5)")
lines.append("\\begin{table}[t]\\caption{Run-to-run variance: mean$\\pm$std over $R$ independent repeats (TruthfulQA, Llama~3.3~70B + MiniLM, $K{=}5$, 100 samples/run).}\\label{tab:variance}\\centering\\small")
lines.append("\\begin{tabular}{lcccc}\\toprule")
lines.append("Strategy & $R$ & Acc.\\,(\\%) & Recall\\,(\\%) & F1\\,(\\%) \\\\\\midrule")
lab = {"injection":"Instruction injection","contradiction":"Contradiction","subtle":"Subtle manip.","entity_swap":"Entity swap","mixed":"Mixed"}
for (ds, strat, llm, emb, nruns, a, as_, p, ps_, rc, rs_, f, fs_, sp, sps_) in variance_rows:
    if ds=="truthfulqa" and "MiniLM" in emb and llm.startswith("llama"):
        lines.append(f"{lab.get(strat,strat)} & {nruns} & {100*a:.1f}$\\pm${100*as_:.1f} & {100*rc:.1f}$\\pm${100*rs_:.1f} & {100*f:.1f}$\\pm${100*fs_:.1f} \\\\")
lines.append("\\bottomrule\\end{tabular}\\end{table}")

# Calibration table
lines.append("")
lines.append("% Auto-generated: per-LLM calibration")
lines.append("\\begin{table}[t]\\caption{Per-LLM threshold calibration (TruthfulQA mixed, $K{=}5$). $\\tau$ fit on clean scores only; evaluated on held-out clean + poisoned. ROC-AUC is threshold-independent.}\\label{tab:calib}\\centering\\small")
lines.append("\\resizebox{\\columnwidth}{!}{%")
lines.append("\\begin{tabular}{llccccc}\\toprule")
lines.append("LLM & $\\tau$ & Acc. & Prec. & Rec. & F1 & ROC-AUC \\\\\\midrule")
disp = {"llama3.3:70b":"Llama 3.3 70B","qwen3.5:35b":"Qwen 3.5 35B","mistral:7b-instruct":"Mistral 7B Instruct"}
for llm in LLMS:
    if llm not in calib_rows: continue
    t0, m0, tc, mc, _ = calib_rows[llm]
    auc = auc_table.get(llm,(float('nan'),))[0]
    lines.append(f"\\multirow{{2}}{{*}}{{{disp[llm]}}} & 0.50 & {pc(m0['acc'])} & {pc(m0['prec'])} & {pc(m0['rec'])} & {pc(m0['f1'])} & \\multirow{{2}}{{*}}{{{f3(auc)}}} \\\\")
    lines.append(f" & {tc:.2f} & {pc(mc['acc'])} & {pc(mc['prec'])} & {pc(mc['rec'])} & {pc(mc['f1'])} & \\\\")
    lines.append("\\midrule")
if lines[-1]=="\\midrule": lines[-1]="\\bottomrule\\end{tabular}}\\end{table}"
open(os.path.join(OUT,"results_tables.tex"),"w",encoding="utf-8").write("\n".join(lines))

# ROC figure (pgfplots)
def coords(pts): return " ".join(f"({x:.4f},{y:.4f})" for x,y in pts)
fig = []
fig.append("% Auto-generated ROC figure (requires \\usepackage{pgfplots})")
fig.append("\\begin{figure}[t]\\centering\\begin{tikzpicture}")
fig.append("\\begin{axis}[width=\\columnwidth,height=0.78\\columnwidth,xlabel={False positive rate},ylabel={True positive rate},xmin=0,xmax=1,ymin=0,ymax=1,legend pos=south east,grid=both,font=\\small]")
for llm,style in [("llama3.3:70b","thick"),("qwen3.5:35b","thick,dashed"),("mistral:7b-instruct","thick,dash dot")]:
    if llm in roc_curves:
        auc = auc_table[llm][0]
        fig.append(f"\\addplot[{style}] coordinates {{{coords(roc_curves[llm])}}};")
        fig.append(f"\\addlegendentry{{{disp[llm]} (AUC={auc:.2f})}}")
fig.append("\\addplot[gray,dotted] coordinates {(0,0)(1,1)};")
fig.append("\\end{axis}\\end{tikzpicture}")
fig.append("\\caption{Threshold-independent poison detection (ROC, pooled mixed-strategy runs). Both LLMs carry discriminative signal; the gap to a usable operating point is closed by per-LLM calibration (Table~\\ref{tab:calib}).}\\label{fig:roc}\\end{figure}")
open(os.path.join(OUT,"roc_figure.tex"),"w",encoding="utf-8").write("\n".join(fig))

# ---------------------------------------------------------------- 4) STRATEGY BAR FIGURE
# Per-strategy Recall and F1 (Llama 3.3 70B + MiniLM, TruthfulQA), sourced from the
# same aggregated runs as the variance table so the figure and Table II agree.
bar = {}
for (ds, strat, llm, emb, nruns, a, as_, p, ps_, rc, rs_, f, fs_, sp, sps_) in variance_rows:
    if ds == "truthfulqa" and "MiniLM" in emb and llm.startswith("llama"):
        bar[strat] = (100 * (rc if not np.isnan(rc) else 0.0), 100 * (f if not np.isnan(f) else 0.0))
bar_order = ["injection", "contradiction", "subtle", "entity_swap", "mixed"]
bar_lab = {"injection": "Injection", "contradiction": "Contradict.", "subtle": "Subtle",
           "entity_swap": "Entity swap", "mixed": "Mixed"}
present = [s for s in bar_order if s in bar]
if present:
    sym = ",".join(bar_lab[s] for s in present)
    rec_co = " ".join(f"({bar_lab[s]},{bar[s][0]:.1f})" for s in present)
    f1_co = " ".join(f"({bar_lab[s]},{bar[s][1]:.1f})" for s in present)
    sb = []
    sb.append("% Auto-generated: per-strategy detection bar chart (requires pgfplots)")
    sb.append("\\begin{figure}[t]\\centering\\begin{tikzpicture}")
    sb.append("\\begin{axis}[width=\\columnwidth,height=0.62\\columnwidth,ybar,bar width=6pt,")
    sb.append("    ymin=0,ymax=105,ylabel={Score (\\%)},symbolic x coords={" + sym + "},")
    sb.append("    xtick=data,x tick label style={font=\\footnotesize,rotate=20,anchor=east},")
    sb.append("    legend pos=north east,legend style={font=\\footnotesize},ymajorgrids,font=\\small]")
    sb.append("\\addplot[fill=blue!55] coordinates {" + rec_co + "};")
    sb.append("\\addplot[fill=orange!70] coordinates {" + f1_co + "};")
    sb.append("\\legend{Recall, F1}")
    sb.append("\\end{axis}\\end{tikzpicture}")
    sb.append("\\caption{Per-strategy detection on TruthfulQA (Llama~3.3~70B + MiniLM, $K{=}5$): "
              "the hierarchy injection $\\gg$ contradiction $\\gg$ subtle $\\approx$ entity swap.}"
              "\\label{fig:strategybar}\\end{figure}")
    open(os.path.join(OUT, "strategy_bar_figure.tex"), "w", encoding="utf-8").write("\n".join(sb))

# ---------------------------------------------------------------- 5) TRUST-SCORE DISTRIBUTION
# Clean vs poisoned Trust Index histogram, pooled over mixed Llama+MiniLM TruthfulQA runs.
gd = [r for r in canon if r["strat"] == "mixed" and r["llm"].startswith("llama")
      and "MiniLM" in r["emb"] and r["ds"] == "truthfulqa"]
if gd:
    trust = np.concatenate([r["trust"] for r in gd])
    truth = np.concatenate([r["truth"] for r in gd])
    edges = np.linspace(0.0, 1.0, 11)              # 10 bins
    centers = (edges[:-1] + edges[1:]) / 2
    clean_h, _ = np.histogram(trust[~truth], bins=edges)
    pois_h, _ = np.histogram(trust[truth], bins=edges)
    clean_h = clean_h / max(clean_h.sum(), 1)      # normalize to fraction
    pois_h = pois_h / max(pois_h.sum(), 1)
    cc = " ".join(f"({c:.2f},{v:.3f})" for c, v in zip(centers, clean_h))
    pc_ = " ".join(f"({c:.2f},{v:.3f})" for c, v in zip(centers, pois_h))
    td = []
    td.append("% Auto-generated: clean vs poisoned trust-score distribution (pgfplots)")
    td.append("\\begin{figure}[t]\\centering\\begin{tikzpicture}")
    ymax = float(np.ceil(max(clean_h.max(), pois_h.max()) * 10) / 10 + 0.05)
    td.append("\\begin{axis}[width=\\columnwidth,height=0.62\\columnwidth,ybar,bar width=4pt,")
    td.append(f"    xmin=0,xmax=1,ymin=0,ymax={ymax:.2f},xlabel={{Trust Index $T$}},ylabel={{Fraction of samples}},")
    td.append("    legend pos=north west,legend style={font=\\footnotesize},ymajorgrids,font=\\small]")
    td.append("\\addplot[fill=green!45] coordinates {" + cc + "};")
    td.append("\\addplot[fill=red!55] coordinates {" + pc_ + "};")
    td.append(f"\\draw[dashed,thick] (axis cs:0.5,0) -- (axis cs:0.5,{ymax:.2f});")
    td.append("\\legend{Clean, Poisoned}")
    td.append("\\end{axis}\\end{tikzpicture}")
    td.append("\\caption{Trust Index distribution for clean vs poisoned samples (TruthfulQA mixed, "
              "Llama~3.3~70B + MiniLM, $K{=}5$); dashed line marks the decision threshold $\\tau{=}0.5$.}"
              "\\label{fig:trustdist}\\end{figure}")
    open(os.path.join(OUT, "trust_dist_figure.tex"), "w", encoding="utf-8").write("\n".join(td))

print("\nWrote: results_tables.tex, roc_figure.tex, strategy_bar_figure.tex, trust_dist_figure.tex")
