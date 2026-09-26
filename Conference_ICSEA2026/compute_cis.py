"""Compute Wilson and bootstrap 95% confidence intervals for the headline
detection metrics, from the confusion-matrix counts reported in the paper.
No new model inference is required: the metrics depend
only on the TP/FP/FN/TN counts, so resampling reconstructed per-sample outcomes
is statistically identical to resampling the original per-sample binary outcomes.

Run:  python paper/compute_cis.py
"""

import math
import random

random.seed(42)
Z = 1.96
B = 20000  # bootstrap resamples


def wilson(k, n, z=Z):
    """Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return (center - margin, center + margin)


def f1_from_counts(tp, fp, fn):
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)


def bootstrap_metric(tp, fp, fn, tn, metric):
    """Percentile bootstrap CI by resampling the reconstructed per-sample set."""
    samples = (
        [("pos", "flag")] * tp
        + [("pos", "noflag")] * fn
        + [("neg", "flag")] * fp
        + [("neg", "noflag")] * tn
    )
    n = len(samples)
    vals = []
    for _ in range(B):
        rs = [samples[random.randrange(n)] for _ in range(n)]
        rtp = sum(1 for t, p in rs if t == "pos" and p == "flag")
        rfp = sum(1 for t, p in rs if t == "neg" and p == "flag")
        rfn = sum(1 for t, p in rs if t == "pos" and p == "noflag")
        rtn = sum(1 for t, p in rs if t == "neg" and p == "noflag")
        if metric == "f1":
            vals.append(f1_from_counts(rtp, rfp, rfn))
        elif metric == "acc":
            vals.append((rtp + rtn) / n)
    vals.sort()
    lo = vals[int(0.025 * B)]
    hi = vals[int(0.975 * B)]
    return lo, hi


def pct(x):
    return f"{100 * x:.1f}%"


# Confusion counts (TP, FP, FN, TN); each run is N=100 (15 poisoned, 85 clean
# for runs with FP=0); contradiction/injection have 84 TN due to 1 FP.
runs = {
    "Mixed (K=5, primary)": dict(tp=6, fp=0, fn=9, tn=85),
    "Contradiction":        dict(tp=8, fp=1, fn=7, tn=84),
    "Injection":            dict(tp=15, fp=1, fn=0, tn=84),
    "Entity Swap":          dict(tp=0, fp=0, fn=15, tn=85),
    "Subtle":               dict(tp=3, fp=0, fn=12, tn=85),
}

print(f"{'Run':<24} {'Acc [95% CI]':<26} {'Recall [95% CI]':<26} {'F1 [95% CI]'}")
print("-" * 100)
for name, c in runs.items():
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    n = tp + fp + fn + tn
    acc = (tp + tn) / n
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = f1_from_counts(tp, fp, fn)
    acc_lo, acc_hi = wilson(tp + tn, n)
    rec_lo, rec_hi = wilson(tp, tp + fn)
    f1_lo, f1_hi = bootstrap_metric(tp, fp, fn, tn, "f1")
    print(
        f"{name:<24} "
        f"{pct(acc)+' ['+pct(acc_lo)+'-'+pct(acc_hi)+']':<26} "
        f"{pct(rec)+' ['+pct(rec_lo)+'-'+pct(rec_hi)+']':<26} "
        f"{pct(f1)+' ['+pct(f1_lo)+'-'+pct(f1_hi)+']'}"
    )
