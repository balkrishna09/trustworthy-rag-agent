# Conference Paper — ICSEA 2026

Workspace for turning the master's thesis into a conference paper.

## Target venue (locked)

- **ICSEA 2026** — 21st Int'l Conference on Software Engineering Advances (IARIA)
- **Rank:** CORE C (lowest tier, as requested) · non-workshop · European
- **Location:** Barcelona, Spain — *virtual presentation allowed* (no travel needed)
- **Paper deadline:** **June 26, 2026** · Notification: Jul 26 · Camera-ready: Aug 23 · Conference: Sep 27 – Oct 1, 2026
- **Length:** 6 pages incl. references (up to +4 pages at extra cost) · IEEE-style 2-column

## Files

| File | Purpose |
|---|---|
| `icsea2026_paper.tex` | The paper draft (IEEEtran). `\input`s the two generated snippets below. **Compile on Overleaf** (upload the whole folder). ~7–8 pages with the robustness section. |
| `robustness_analysis.py` | Computes run-to-run variance, ROC/AUC, and per-LLM calibration from stored per-sample results (no LLM). Regenerates the two `.tex` snippets below. |
| `results_tables.tex` | **Generated** — variance + calibration tables. |
| `roc_figure.tex` | **Generated** — ROC figure (pgfplots). |
| `compute_cis.py` | Computes Wilson + bootstrap 95% CIs from confusion counts (no model runs). |
| `SUPERVISOR_EMAIL.md` | Ready-to-send email to get supervisor sign-off, co-authorship, and fee funding. |
| `README.md` | This file. |

## Status

- [x] Readiness assessed: **conference-ready for CORE C** — no future-work items required to submit.
- [x] Full draft written by condensing the thesis (intro, related work, method, setup, results, discussion, conclusion).
- [x] Confidence intervals computed and embedded (the "light strengthening" achievable now).
- [x] **Robustness section added** (no LLM needed): multi-run variance (90.6%±0.5% over 5 repeats), ROC/AUC (Llama 0.81 / Qwen 0.73), and per-LLM calibration that recovers Qwen (65.5%→74.5%). See `robustness_analysis.py`.
- [ ] **Supervisor sign-off** (send `SUPERVISOR_EMAIL.md`) — fill in name/email first.
- [ ] Fill placeholders in `icsea2026_paper.tex` (author block, supervisor).
- [ ] Confirm official IARIA template vs. IEEEtran; adjust if needed.
- [ ] Compile (Overleaf), check page count (~7–8 pages now; trim to 6 if avoiding extra-page fees), proofread.
- [ ] Submit via the IARIA system by **June 26, 2026**.

## How to compile

**Option A — Overleaf (recommended, no install):** create a new project, upload
the whole `Conference_ICSEA2026` folder (the paper `\input`s `results_tables.tex`
and `roc_figure.tex`), set compiler to *pdfLaTeX*, and compile twice. Uses
`pgfplots`; bibliography is embedded (no `.bib` needed).

**Option B — Local:** install MiKTeX, then `pdflatex` the file twice (run from
inside the folder so the `\input`s resolve).

## Reproduce / regenerate the analysis

```
python Conference_ICSEA2026/compute_cis.py            # confidence intervals
python Conference_ICSEA2026/robustness_analysis.py    # variance, ROC/AUC, calibration -> regenerates the .tex snippets
```

## Camera-ready strengthening (only if accepted — Jul 26 → Aug 23 window)

1. One extra baseline beyond naive (RAGAS-style faithfulness or retrieval-confidence).
2. Poison-signal ablation (disable each of the 5 signals; report ΔF1).
3. Optional: 3–5 repeated runs for true run-to-run variance.
