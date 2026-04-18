# Master's Thesis Summary
## Submitted in Support of PhD Application — Chalmers University of Technology
### Position: Doctoral Student in Network Security (REF 2025-0599)

---

**Candidate:** Balkrishna Giri
**Current Institution:** Tampere University (TUNI), Finland
**Degree Program:** Master of Science (Technology) — Computing and Electrical Engineering
**Expected Graduation:** Summer 2026
**Thesis Status:** In Progress (~75% complete) — all chapters under supervisor review
**Supervisor Meeting:** March 5, 2026

---

## Thesis Title

**Trustworthy RAG: An Evaluation Agent for Detecting Misinformation and Knowledge Poisoning in Generative AI Systems**

---

## Abstract

Large Language Models (LLMs) augmented with external knowledge bases through Retrieval-Augmented Generation (RAG) have become the dominant paradigm for deploying AI in information-critical domains. However, standard RAG architectures contain a fundamental security-reliability gap: they assume the external knowledge base is benign and trustworthy. An adversary who inserts malicious documents into the retrieval corpus can silently manipulate the model's outputs — a threat known as *knowledge poisoning* or *indirect prompt injection*. This is structurally analogous to adversarial data injection in distributed systems: just as a Byzantine node can corrupt consensus in a peer-to-peer network by broadcasting false state, a poisoned document can corrupt an LLM's reasoning by injecting false context into the retrieval layer.

This thesis designs, implements, and empirically evaluates a **Trustworthy RAG Framework** — an autonomous Evaluation Agent that acts as a defensive middleware layer within the RAG pipeline. The agent intercepts retrieved documents before generation and assigns each response a quantitative **Trust Index** based on three dimensions: factual consistency (via Natural Language Inference), cross-document consistency, and poisoning probability. Responses falling below a configurable threshold trigger an actionable warning, enabling downstream systems to reject or re-query the compromised context. The work contributes both a principled security metric and an empirical characterisation of attack detectability across four adversarial strategies.

---

## Research Questions

| | Question |
|---|---|
| **RQ1** | What computational strategies (NLI, semantic discrepancy detection) are effective for detecting misinformation and knowledge poisoning in retrieved contexts? |
| **RQ2** | How can a composite Trust Index be mathematically formulated to quantify the reliability of a RAG response? |
| **RQ3** | To what extent does an Evaluation Agent enhance RAG security compared to a baseline, and what are the trade-offs in latency? |

---

## System Architecture

The Evaluation Agent is composed of three modules operating in sequence:

1. **NLI Verifier** — Uses `facebook/bart-large-mnli` (Natural Language Inference) to assess entailment between the generated response and each retrieved document. Documents with entailment probability > 0.4 contribute to the factuality score; contradiction-only documents are excluded from factuality and flagged to the poisoning detector instead.

2. **Poison Detector** — Multi-signal detection combining: (a) intra-document NLI (first-half vs. second-half self-contradiction as a signal of injected adversarial payloads), (b) cross-document contradiction with relevance-weighted aggregation, (c) pattern-based suspicious phrase detection, and (d) structural anomaly analysis. Detection scores are weighted by FAISS retrieval confidence so low-relevance neighbours do not inflate the overall risk — analogous to discounting low-trust peers in distributed reputation systems.

3. **Trust Index Calculator** — Aggregates the three dimensions into a composite score:

   **T = 0.40 · Factuality + 0.35 · Consistency + 0.25 · (1 − Poison Probability)**

   A non-linear dampener applies additional penalty when `poison_probability > 0.70`, preventing high factuality from masking confirmed poisoning signals — a deliberate design choice to prioritize security recall over trust score inflation.

---

## Key Experimental Results

Experiments were conducted on two public benchmarks (TruthfulQA, FEVER) against four adversarial poisoning strategies: contradiction injection, instruction injection, entity swap, and subtle manipulation.

### Overall Detection Performance (TruthfulQA, 50-sample mixed experiment)

| Metric | Value |
|--------|-------|
| System Accuracy | 86% |
| Precision | 52.4% |
| Recall | 73.3% |
| F1 Score | 61.1% |
| Trust Score Separation (Δ) | 0.263 |
| Improvement over Naive Baseline | +1% accuracy |

### Per-Strategy Detection (20 samples each, TruthfulQA)

| Poisoning Strategy | Accuracy | Precision | Recall | F1 | Separation |
|--------------------|----------|-----------|--------|----|------------|
| Contradiction Injection | 85% | 50.0% | 33.3% | 40.0% | 0.245 |
| Instruction Injection | 75% | 37.5% | **100%** | 54.5% | **0.407** |
| Entity Swap | 87.5% | **100%** | 16.7% | 28.6% | 0.071 |
| Subtle Manipulation | 85% | 50.0% | 16.7% | 25.0% | 0.127 |
| Mixed Strategies | 87.5% | 55.6% | 83.3% | 66.7% | 0.327 |

### Factuality Improvement (FEVER benchmark)

| Metric | Value |
|--------|-------|
| Average Trust Score (Clean KB) | 0.654 |
| System Accuracy | **90%** |
| Improvement over Baseline | **+5%** |

---

## Novel Contributions

1. **Adversarial-aware Trust Index** — A principled composite metric combining NLI factuality, cross-document consistency, and poison probability; novel in its integration of a non-linear dampener that prevents masking of confirmed poisoning signals under high-factuality conditions.

2. **Intra-document NLI poisoning signal** — Detection of self-contradicting documents (first half vs. second half) as a lightweight, training-free method for flagging contradiction-injection attacks with no labelled training data required.

3. **Relevance-weighted poison aggregation** — Weighting per-document poison scores by retrieval confidence before computing batch-level risk, significantly reducing false positives caused by low-relevance poisoned neighbours — directly analogous to confidence-weighted peer voting in distributed trust protocols.

4. **Empirical characterisation of attack detectability** — A systematic comparison across four attack strategies demonstrating that instruction injection is detectable at 100% recall, while entity swap and subtle manipulation represent fundamental architectural limits of NLI-based analysis — providing clear direction for future defense research.

5. **Identification of the RAGAS/ARES blind spot** — Formal demonstration that context-faithfulness metrics (the dominant RAG evaluation paradigm) are orthogonal to context-integrity verification, motivating a new security-focused evaluation dimension for AI information pipelines.

---

## Current Thesis Status

| Chapter | Title | Status |
|---------|-------|--------|
| Chapter 1 | Introduction | Under Review |
| Chapter 2 | Theoretical Background | Under Review |
| Chapter 3 | Literature Review | Under Review |
| Chapter 4 | Research Methodology | Under Review |
| Chapter 5 | Evaluation Agent Implementation | Under Review |
| Chapter 6 | Experimental Evaluation | Under Review |
| Chapter 7 | Discussion | Under Review |
| Chapter 8 | Conclusion & Future Work | Under Review |

All experimental work is finalized. The full thesis draft is currently under supervisor review ahead of the scheduled meeting on March 5, 2026.

---

## Relevance to REF 2025-0599 — Network Security in Distributed Systems

The Chalmers doctoral position focuses on network security challenges in distributed and decentralized infrastructures, including vulnerability analysis, attack surface characterization, and defense protocol design. My thesis addresses the same class of problems at the **application layer of AI-powered information retrieval systems**, with several direct bridges to the position's research agenda:

**1. Adversarial Data Injection as a Distributed Systems Attack**
Knowledge poisoning in RAG is structurally equivalent to Byzantine data injection in peer-to-peer networks. In both cases, an adversarial node/document is optimized to be semantically indistinguishable from legitimate data (high cosine similarity to honest content), yet carries a malicious payload. My work demonstrates that multi-signal detection — analogous to cross-peer verification in Byzantine fault-tolerant consensus — is more effective than single-metric approaches, achieving 66.7% F1 in mixed attack scenarios.

**2. Attack Surface Characterization Methodology**
The per-strategy experimental design (evaluating four attack types in isolation and in combination) directly mirrors the vulnerability analysis methodology applied to network protocols and consensus mechanisms. The finding that different attack strategies have systematically different detectability profiles (injection: 100% recall; entity swap: 16.7% recall) is the kind of attack surface characterization that informs targeted defense design in distributed security research.

**3. Distributed Trust and Reputation Mechanisms**
The Trust Index architecture — where multiple independent signals are aggregated with confidence weighting to produce a single reliability score — directly parallels distributed reputation systems and trust management protocols studied in network security. The non-linear dampener (penalizing confirmed poisoning even when factuality is high) is equivalent to a veto mechanism in consensus: a single strong negative signal can override aggregate positive votes.

**4. Protocol-Level Defense Without Retraining**
A key finding is that effective detection is achievable without modifying the underlying LLM or retrieval model — the defense operates purely at the protocol/middleware layer. This aligns with the network security principle of building security into the communication infrastructure rather than depending on endpoint trust, which is particularly relevant to securing decentralized AI systems and federated RAG deployments.

**5. Scalable Defense Design**
The relevance-weighted aggregation mechanism reduces false positive rates as the number of retrieved documents grows — a scalability property that directly maps to securing larger peer-to-peer networks where the ratio of adversarial to honest nodes may vary.

I am strongly motivated to extend this research toward the specific challenges of securing AI systems in decentralized and blockchain-adjacent environments — areas where data provenance, tamper-evidence, and peer verification are fundamental unsolved problems at the intersection of network security and AI trustworthiness.

---

*For further information or access to the working thesis draft and codebase, please contact: balkrishna.giri@tuni.fi*
