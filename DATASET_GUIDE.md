# Dataset Guide for Trustworthy RAG Project

## Overview

Since datasets won't be provided by your supervisor, here's how to obtain and prepare them yourself.

---

## 📊 Required Datasets

### 1. **FEVER Dataset** (Fact Extraction and VERification)
- **Purpose**: Evaluate factual verification capabilities (RQ1, RQ2)
- **Size**: ~1.5GB
- **Source**: Publicly available
- **Download**: Via Hugging Face Datasets (easiest method)

### 2. **TruthfulQA Dataset**
- **Purpose**: Evaluate misinformation detection (RQ1, RQ2)
- **Size**: ~50MB
- **Source**: Publicly available
- **Download**: Via Hugging Face Datasets

### 3. **Poisoned Dataset** (To be created)
- **Purpose**: Knowledge poisoning attack evaluation (RQ1, RQ3)
- **Size**: Variable (depends on your experiments)
- **Source**: You'll create this yourself
- **Method**: Inject adversarial content into clean datasets

---

## 🎯 Recommended Approach

### **Option 1: Hugging Face Datasets (RECOMMENDED)**

**Why this is best:**
- ✅ Already have `datasets` library installed
- ✅ Automatic download and caching
- ✅ Pre-processed and ready to use
- ✅ No manual file management
- ✅ Standard format (easy to work with)

**How it works:**
- Datasets are downloaded automatically when you load them
- Stored in `~/.cache/huggingface/datasets/`
- Can be loaded directly in Python code

### **Option 2: Direct Download (Alternative)**

If Hugging Face doesn't have the exact dataset:
- FEVER: https://fever.ai/dataset/fever.html
- TruthfulQA: https://github.com/sylinrl/TruthfulQA

---

## 📥 Dataset Download Script

I've created `download_datasets.py` that will:
1. Download FEVER dataset from Hugging Face
2. Download TruthfulQA dataset from Hugging Face
3. Save them to `data/raw/` directory
4. Show basic statistics

**Run it:**
```powershell
python download_datasets.py
```

---

## 🧪 Creating Poisoned Dataset

### Strategy for Knowledge Poisoning:

1. **Start with clean dataset** (FEVER or custom)
2. **Inject adversarial content**:
   - **Factual manipulation**: Change facts in documents
   - **Semantic injection**: Add misleading but plausible information
   - **Context poisoning**: Add false context that seems relevant
3. **Label poisoned samples**: Mark which documents are poisoned
4. **Create test queries**: Queries that should trigger poisoned content

### Example Poisoning Techniques:

**Type 1: Direct Fact Manipulation**
```
Original: "Paris is the capital of France."
Poisoned: "London is the capital of France."
```

**Type 2: Context Injection**
```
Original: "Python is a programming language."
Poisoned: "Python is a programming language. However, Python 4.0 was released in 2023 with major breaking changes."
```

**Type 3: Semantic Similarity Attack**
```
Original: "The Earth orbits the Sun."
Poisoned: "The Earth orbits the Sun. However, recent studies suggest the Sun orbits the Earth."
```

### When to Create:
- After you have your RAG pipeline working
- Use a subset of FEVER or create custom documents
- Inject 10-30% poisoned content for testing

---

## 📁 Dataset Structure

After downloading, your `data/` directory will look like:

```
data/
├── raw/
│   ├── fever/              # FEVER dataset files
│   └── truthfulqa/         # TruthfulQA dataset files
├── processed/
│   ├── fever_processed.json
│   └── truthfulqa_processed.json
└── poisoned/
    ├── poisoned_fever.json
    └── poisoned_custom.json
```

---

## 🔍 Dataset Details

### FEVER Dataset
- **Format**: JSON Lines (.jsonl)
- **Fields**: 
  - `claim`: The statement to verify
  - `label`: SUPPORTS, REFUTES, or NOT ENOUGH INFO
  - `evidence`: Supporting documents
- **Use**: Test factual verification (NLI verifier)

### TruthfulQA Dataset
- **Format**: JSON/CSV
- **Fields**:
  - `Question`: Question to answer
  - `Best Answer`: Correct answer
  - `Incorrect Answers`: Misleading answers
- **Use**: Test misinformation detection

---

## ✅ Next Steps

1. **Download datasets** using `download_datasets.py`
2. **Explore datasets** in a Jupyter notebook
3. **Create poisoned dataset** after pipeline is ready
4. **Use for evaluation** in your experiments

---

## 📚 Additional Resources

- **Hugging Face Datasets**: https://huggingface.co/datasets
- **FEVER Dataset**: https://fever.ai/
- **TruthfulQA Paper**: https://arxiv.org/abs/2109.07958

---

**Note**: All datasets mentioned are publicly available for research purposes. No special permissions needed.
