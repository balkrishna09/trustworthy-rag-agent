# Thesis Accuracy Review — Trustworthy RAG Agent
**Author**: Balkrishna Giri
**Last Updated**: 2026-03-09
**Source reviewed**: All `.tex` files in `Master_Thesis/chapters/` + `main.tex`
**Cross-referenced against**: Complete codebase (`src/evaluation_agent/`, `src/experiments/`)

---

## How to Use This Document

This document is the **single source of truth** for thesis corrections. Each issue has:
- The exact file and location
- What the thesis currently says
- What the code actually does
- The exact fix to apply

Severity levels:
- **CRITICAL** — Chapter is empty or will fail review
- **HIGH** — Factual inaccuracy or major missing explanation
- **MEDIUM** — Missing formula/detail that weakens reproducibility
- **LOW** — Minor clarification

Status markers:
- ✅ **FIXED** — Confirmed correct in latest `.tex` source
- ❌ **NEEDS FIX** — Issue confirmed in `.tex` source
- 📝 **NEEDS WRITE** — Section is an empty stub

---

## Chapter Status Overview

| File | Status | Notes |
|------|--------|-------|
| `main.tex` (abstract) | ❌ Empty | Write abstract (text drafted in Part 1 below) |
| `ch1_introduction.tex` | ✅ Complete | Updated 2026-03-07 |
| `ch2_theory_background.tex` | ✅ Complete | Updated 2026-03-07 |
| `ch3_literature_review.tex` | ✅ Complete | Updated 2026-03-07 |
| `ch4_methodology.tex` | ✅ Complete | All 6 gaps fixed 2026-03-09 |
| `ch5_eval_agent.tex` | ⚠️ Mostly complete | 1 low-priority gap remaining (Issue 3.4) |
| `ch6_experiments.tex` | ✅ Complete | Sec 6.6 + Sec 6.7 added; K=5 results documented |
| `ch7_discussion.tex` | 📝 Empty stub | Full chapter needed |
| `ch8_conclusion.tex` | 📝 Empty stub | Full chapter needed |

---

## Previously Identified Issues — Confirmed FIXED in .tex Source

| Issue | Old value / state | Fixed to | Date |
|-------|-------------------|----------|------|
| Trust Index formula | `w₁·S_source − w₃·P_poison` | `α·S_factuality + β·S_consistency + γ·(1−P_poison)` with α=0.4, β=0.35, γ=0.25 | earlier |
| Hardware (A100 claim) | "NVIDIA A100 GPU" | "FARMI computing cluster; NLI runs on CPU" | earlier |
| Poisoning technique | Referenced Zou et al. gradient method | 4 rule-based strategies correctly described | earlier |
| Poisoning rate | "1%, 5% of retrieved context" | "30% poison ratio" | earlier |
| Relevance-weighted aggregation | Not mentioned | Eq. 5.2 with formula | earlier |
| Ch1 scope (single LLM) | "Llama 3.3 70B" only | Both LLMs + 2×2 factorial design | 2026-03-07 |
| Ch1 RQ3 | No model sensitivity | Extended to include LLM/embedding effect | 2026-03-07 |
| Ch2 dense retrieval | MiniLM only | Added Snowflake note + fwd ref to Ch6 | 2026-03-07 |
| Ch3 research gaps | 3 gaps | Added 4th: LLM generation-style sensitivity | 2026-03-07 |
| Ch4 components | Llama/MiniLM only | Both LLMs, both embeddings; Multi-Model subsection | 2026-03-07 |
| Ch5 LLM-dependency | Not mentioned | Added `\subsubsection{LLM-Dependency of the Trust Index}` | 2026-03-07 |
| Ch6 Table 6.1 | 0.824/0.561/0.263/86%/+1% | 0.830/0.590/0.240/91%/+7% | 2026-03-07 |
| Ch6 Table 6.2 | 86%/52.4%/73.3%/61.1% | 91%/100%/40%/57.1% | 2026-03-07 |
| Ch6 Section 6.6 | Missing | Full 2×2 grid table + 3 findings + deployment implications | 2026-03-07 |
| **Ch4 dampener formula (Issue 2.1)** | Mentioned in prose only | Added δ equation + boundary values + fwd ref to Ch5 | **2026-03-09** |
| **Ch4 Entity Swap fallback (Issue 2.2)** | In-place replacement only | 3-level fallback chain fully described | **2026-03-09** |
| **Ch4 ASR→PDR (Issue 2.3)** | ASR only, misaligned with code | PDR (primary metric) + ASR caveat explaining why true ASR unmeasured | **2026-03-09** |
| **Ch4 K=5→K=3 in grid subsection** | Said K=5 for 2×2 grid | Corrected to K=3; added fwd ref to Sec 6.7 | **2026-03-09** |
| **Ch5 BART-MNLI ~0.99 (Issue 3.1)** | Entailment strategy stated without motivation | Added paragraph: MNLI gives ~0.99 contradiction to unrelated pairs → entailment-only design | **2026-03-09** |
| **Ch5 FAISS threshold (Issue 3.3)** | "weak top score can reduce trust" | Explicit `$s_{\text{best}} < 0.30$` threshold with eq ref | **2026-03-09** |
| **Ch6 Sec 6.6 K=5→K=3** | Grid description said K=5 | Corrected to K=3; added cross-ref to Sec 6.7 | **2026-03-09** |
| **Ch6 Section 6.7 (K sensitivity)** | TBD placeholder | Fully written with K=5 results, null-result finding, architectural implication | **2026-03-09** |

---

## Ground Truth Experiment Data

### 2×2 Grid Results (K=3, validated)

All four runs: 100 samples (50 clean Part A + 50 poisoned Part B), mixed strategy, 30% poison ratio, **K=3**, τ=0.5.

| Configuration | Acc | Prec | Recall | F1 | Clean T̄ | Poison T̄ | Sep | TP | FP | FN | vs Baseline |
|---------------|-----|------|--------|----|---------|---------|-----|----|----|----|----|
| Llama 3.3 70B + all-MiniLM-L6-v2 | 91% | 100% | 40% | 57.1% | 0.830 | 0.590 | 0.240 | 6 | 0 | 9 | +7% |
| Llama 3.3 70B + snowflake-arctic-embed2 | 91% | 100% | 40% | 57.1% | 0.799 | 0.611 | 0.188 | 6 | 0 | 9 | +7% |
| Qwen 3.5 35B + all-MiniLM-L6-v2 | 71% | 25.0% | 46.7% | 32.6% | 0.633 | 0.471 | 0.161 | 7 | 21 | 8 | −16% |
| Qwen 3.5 35B + snowflake-arctic-embed2 | 71% | 28.1% | 60.0% | 38.3% | 0.653 | 0.454 | 0.199 | 9 | 23 | 6 | −16% |

Baseline accuracy (naive always-trust): **85%**

### K=5 Sensitivity Run (Llama primary config + all four combos)

All four runs: same setup as grid, **K=5**. Shows K has no effect on Llama detection.

| Configuration | Acc | Prec | Recall | F1 | Clean T̄ | Sep | TP | FP | FN | vs K=3 |
|---------------|-----|------|--------|----|---------|-----|----|----|----|----|
| Llama 3.3 70B + all-MiniLM-L6-v2 | 91% | 100% | 40% | 57.1% | 0.827 | 0.237 | 6 | 0 | 9 | **no change** |
| Llama 3.3 70B + snowflake-arctic-embed2 | 91% | 100% | 40% | 57.1% | 0.799 | 0.220 | 6 | 0 | 9 | **no change** |
| Qwen 3.5 35B + all-MiniLM-L6-v2 | 69% | 23.3% | 46.7% | 31.1% | 0.616 | 0.145 | 7 | 23 | 8 | −2% acc |
| Qwen 3.5 35B + snowflake-arctic-embed2 | 69% | 26.5% | 60.0% | 36.7% | 0.645 | 0.191 | 9 | 25 | 6 | −2% acc |

**Key finding (Section 6.7):** K=5 ≡ K=3 for Llama. Detection ceiling is attack-strategy-limited (entity_swap/subtle), not retrieval-depth-limited. K=3 validated as optimal (half the NLI cost, same results).

---

## Remaining Open Issues

### Issue 3.4 — Scoped Cross-Document Signals (LOW) ❌

**File**: `ch5_eval_agent.tex`, cross-document consistency bullet
**Current text**: "pairwise NLI checks among retrieved documents detect inter-document contradictions"
**Missing**: The code attributes signals only to the specific pair (`involved_indices`), preventing spillover to uninvolved clean docs.

**Fix**: Append one sentence: "Contradiction signals are attributed only to the specific document pair involved via scoped signal indices, preventing a single poisoned document from inflating the poison probability of uninvolved clean documents in the same batch."

---

## Part 1: main.tex — Abstract (CRITICAL) 📝

**File**: `main.tex`, lines 58–61. Currently empty.

**Draft abstract** (updated for 2×2 grid + K sensitivity findings):

```
Retrieval-Augmented Generation (RAG) systems improve large language model reliability
by grounding responses in external knowledge. However, RAG architectures inherently
trust their retrieval output, creating a critical Security-Reliability Gap: high
semantic relevance does not guarantee factual truth. Adversaries can exploit this by
injecting malicious documents into the knowledge corpus — a threat known as Knowledge
Poisoning — causing the system to generate targeted misinformation.

This thesis proposes an autonomous Evaluation Agent as a defensive middleware within
the RAG pipeline. The agent combines Natural Language Inference (NLI) for factual
verification, a multi-signal Poison Detector (linguistic, structural, intra-document,
cross-document, and semantic-outlier signals), and a composite Trust Index
(T = 0.4·F + 0.35·C + 0.25·(1−P)) to produce an interpretable trustworthiness
score for each generated response.

Experiments on TruthfulQA and FEVER benchmarks with four adversarial poisoning
strategies (contradiction, instruction injection, entity swap, and subtle manipulation)
demonstrate that the system achieves 91% accuracy and 57.1% F1 score with 100%
precision (zero false positives) using Llama 3.3 70B, a +7% improvement over the
naive always-trust baseline. Instruction injection is detected with 100% recall, while
entity-swap and subtle manipulation remain near-undetectable at the textual level,
representing an architectural limit of surface-signal-based approaches.

A 2×2 factorial experiment (Llama 3.3 70B / Qwen 3.5 35B × all-MiniLM-L6-v2 /
Snowflake Arctic Embed2) reveals that the Trust Index is LLM-dependent: Qwen 3.5 35B's
verbose generation style causes 21–23 false positives on clean content, underperforming
the naive baseline by 16%, while LLM selection dominates embedding model choice as
the key performance factor. A retrieval depth sensitivity analysis (K=3 vs K=5)
confirms that detection performance is retrieval-depth-invariant: the 40% recall
ceiling is set by attack strategy difficulty, not by the number of documents retrieved.

The work addresses a gap left by existing frameworks such as RAGAS and ARES, which
measure faithfulness to context but not the integrity of the context itself.
```

---

## Part 2: ch7_discussion.tex — Full Content Plan (CRITICAL) 📝

### 7.1 Interpretation of Key Findings

**RQ1 answer**: Instruction injection and contradiction are effectively detected via linguistic pattern matching and intra-document NLI (up to 100% recall). Entity swap and subtle manipulation operate below the detection threshold of surface-level signals — representing an architectural limit requiring semantic world-knowledge reasoning.

**RQ2 answer**: The Trust Index T = 0.4·F + 0.35·C + 0.25·(1−P) with non-linear dampener provides a calibrated, interpretable trustworthiness score. The ablation study confirms that weight choice significantly influences the precision-recall trade-off.

**RQ3 answer**: The Evaluation Agent provides +7% accuracy improvement over the naive baseline with Llama 3.3 70B (91% vs 85%), with zero false positives. The 2×2 grid establishes that LLM selection is the dominant factor: Qwen 3.5 35B underperforms the baseline (71%) due to generation-style false positives, requiring per-LLM threshold calibration. The retrieval depth sensitivity analysis shows K=3 is sufficient — detection is bottlenecked by attack strategy, not retrieval quantity.

### 7.2 Contributions

1. **Unified Trust Index**: First composite metric simultaneously aggregating NLI factuality, cross-document consistency, and poisoning probability for RAG evaluation
2. **Multi-signal Poison Detector**: Five-signal combination with relevance-weighted aggregation
3. **Per-strategy empirical analysis**: Characterisation of detection difficulty for four adversarial strategies
4. **Architectural limit identified**: Entity-swap and subtle poisoning are near-undetectable at the textual level
5. **LLM-dependency finding**: First empirical demonstration that NLI-based Trust Index scores are sensitive to LLM generation style
6. **Retrieval-depth invariance**: K=3 and K=5 produce identical detection — detection ceiling is strategy-limited, not retrieval-limited

### 7.3 Limitations

1. Rule-based strategies are simpler than gradient-optimized adversarial attacks
2. Entity swap and subtle manipulation: 16.7% recall — fundamental limit of surface-signal detection
3. ~13s/sample latency — impractical for real-time applications
4. Fixed trust threshold τ=0.5 is not adaptive to domain, LLM style, or risk level
5. True ASR not measured (requires semantic comparison of generated answers to injected claims)
6. Per-LLM calibration required: τ=0.5 is tuned for Llama; Qwen needs recalibration

### 7.4 Deployment Implications

- Best suited for **batch, high-stakes contexts** (document verification, regulatory compliance, medical information)
- Injection detection (100% recall) is immediately deployable against prompt injection attacks
- 13s overhead reducible via GPU NLI inference or bounded cross-document checks
- Per-LLM calibration needed before deploying with any LLM other than Llama 3.3 70B

### 7.5 Ethical Alignment

- Interpretable decisions: component scores + warnings align with EU AI Act transparency requirements
- No LLM-as-Judge: evaluation is deterministic (NLI + rules)
- Adversarial dataset generator is a controlled research tool, not an offensive system

---

## Part 3: ch8_conclusion.tex — Full Content Plan (CRITICAL) 📝

### 8.1 Summary of Findings

| Research Question | Finding |
|-------------------|---------|
| RQ1: Detection effectiveness | 100% recall for injection; 40% overall recall; entity-swap/subtle near-undetectable |
| RQ2: Trust Index reliability | T formula + dampener validated by ablation; LLM-dependency requires per-model calibration |
| RQ3: Security improvement vs baseline | +7% over naive baseline (Llama); 13s overhead suitable for batch/high-stakes use |

### 8.2 Contributions

- **Theoretical**: Trust Index mathematical formulation as a 3-dimensional composite metric
- **Theoretical**: Per-strategy detection difficulty characterisation
- **Theoretical**: First empirical demonstration of LLM-dependency and retrieval-depth-invariance in NLI-based trust scoring
- **Practical**: Working implementation integrating FAISS, BART-MNLI, and Llama 3.3 70B with evaluation middleware
- **Practical**: Reproducible experiment framework: 4 attack strategies, 2 benchmarks, 2×2 model grid, K sensitivity

### 8.3 Future Work

1. Gradient-optimized adversarial testing (Zou et al. collision-document approach)
2. Per-LLM Trust Index calibration (replace fixed τ with per-model validated threshold)
3. Semantic world-knowledge integration to detect entity-swap attacks (Wikidata lookup)
4. GPU-accelerated NLI inference (<2s vs current 13s)
5. Multimodal extension (image-text, audio-text RAG)
6. Proper ASR measurement via embedding-similarity comparison
7. Style-normalisation for hedged LLM outputs before NLI inference

---

## Master Quick-Fix Checklist

### Immediate (required before submission):
- [ ] **main.tex**: Write abstract (draft in Part 1 above)
- [ ] **ch7**: Write full chapter (outline in Part 2 above)
- [ ] **ch8**: Write full chapter (outline in Part 3 above)
- [ ] **Appendix stubs**: Create `appA–appE.tex` stub files (or comment out `\include` lines)

### Low-priority polish:
- [ ] **ch5 Issue 3.4**: Add scoped cross-doc signals sentence (see Remaining Open Issues)
- [ ] **ch6**: Add footnote to per-strategy table noting 20-sample vs 100-sample difference
- [ ] **ch6**: Verify section numbering after LaTeX compilation (Section 6.7 + Summary = 6.8 now)

### Already complete ✅:
- [x] ch1 scope + RQ3 extended
- [x] ch2 Snowflake note
- [x] ch3 4th research gap
- [x] ch4 components (both LLMs + embeddings)
- [x] ch4 dampener formula (Issue 2.1)
- [x] ch4 entity swap fallback chain (Issue 2.2)
- [x] ch4 ASR→PDR (Issue 2.3)
- [x] ch4 K=3 corrected in grid subsection
- [x] ch5 LLM-dependency subsubsection
- [x] ch5 BART-MNLI ~0.99 paragraph (Issue 3.1)
- [x] ch5 FAISS threshold explicit (Issue 3.3)
- [x] ch6 tables updated (91%/57.1%/0.240/+7%)
- [x] ch6 Section 6.6 (2×2 grid)
- [x] ch6 Section 6.7 (K=3 vs K=5, fully written)
- [x] run_experiment.py K selection in interactive menu

---

## Appendix Note

`main.tex` includes 5 appendices that don't exist as files yet:
- `chapters/appA_parameters.tex`
- `chapters/appB_results.tex`
- `chapters/appC_prompts.tex`
- `chapters/appD_ethics.tex`
- `chapters/appE_implementation.tex`

These will cause LaTeX compilation errors. Either create stub files or comment out the `\include` lines until ready.

---

*Review v5 — updated 2026-03-09. All ch4/ch5 technical gaps fixed. Ch6 complete with Sections 6.6 and 6.7. K=5 grid experiment run and documented. Remaining work: abstract + ch7 + ch8 + appendix stubs.*
