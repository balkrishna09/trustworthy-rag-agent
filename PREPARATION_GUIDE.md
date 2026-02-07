# Trustworthy RAG - Complete Preparation Guide

## 📋 Overview
This guide covers everything you need to prepare before starting implementation, including what to request from your supervisor.

---

## 🎯 STEP 1: Models & Resources to Request from Supervisor

### **Critical Models Needed:**

#### 1. **LLM for Response Generation** (REQUIRED)
- **Primary Option**: `llama3.3:70b-instruct-q4_K_M` (via Ollama/FARMI)
- **Alternative Options**:
  - `llama3:8b-instruct` (lighter, faster)
  - `mistral:7b-instruct`
  - `llama2:13b-instruct`
- **What to Request**: 
  - Access to FARMI/CSC computing cluster with Ollama service
  - OR GPU access to run models locally
  - OR OpenAI API credits/budget (if using OpenAI)

#### 2. **NLI Model for Factual Verification** (REQUIRED)
- **Primary**: `facebook/bart-large-mnli` (~1.6GB)
- **Alternative**: `microsoft/deberta-large-mnli` (~1.3GB)
- **What to Request**: 
  - Download permission/access to Hugging Face models
  - Storage space (~2-3GB for models)

#### 3. **Embedding Model** (REQUIRED)
- **Primary**: `sentence-transformers/all-MiniLM-L6-v2` (~80MB)
- **Alternative**: `BAAI/bge-m3` (~420MB)
- **What to Request**: 
  - Download permission/access to Hugging Face models
  - Storage space (~500MB)

### **Request Template for Supervisor:**

```
Subject: Model Access Request for Trustworthy RAG Thesis Project

Dear [Supervisor Name],

I need access to the following resources for my Master's thesis:

1. LLM Access:
   - Option A: FARMI/CSC cluster access with Ollama service
   - Option B: GPU compute resources (minimum 16GB VRAM for 70B model)
   - Option C: OpenAI API budget/credits

2. Hugging Face Model Access:
   - NLI Model: facebook/bart-large-mnli (~1.6GB)
   - Embedding Model: sentence-transformers/all-MiniLM-L6-v2 (~80MB)
   - Storage: ~3GB total for models

3. Computing Resources:
   - Minimum: 16GB RAM, 4 CPU cores
   - Recommended: 32GB RAM, 8 CPU cores, GPU access
   - Storage: ~50GB for datasets, models, and experiments

Please let me know which options are available and how to proceed.

Thank you,
[Your Name]
```

---

## 🖥️ STEP 2: Infrastructure Requirements

### **Minimum System Requirements:**
- **RAM**: 16GB (32GB recommended)
- **Storage**: 50GB free space
- **CPU**: 4 cores (8+ recommended)
- **GPU**: Optional but recommended (CUDA-compatible for faster inference)
- **OS**: Windows 10/11, Linux, or macOS

### **Recommended Setup:**
- **For Local Development**: 
  - 32GB RAM
  - NVIDIA GPU with 16GB+ VRAM (for running LLMs locally)
  - SSD storage

- **For Cloud/Cluster Usage**:
  - FARMI/CSC access (preferred)
  - Jupyter Hub access
  - SSH access to compute nodes

---

## 💻 STEP 3: Software Installation Checklist

### **3.1 Python Environment**
```bash
# Check Python version (need 3.10+)
python --version

# If not installed, download from python.org
# Or use Anaconda/Miniconda
```

### **3.2 Git** (if not already installed)
```bash
# Download from: https://git-scm.com/downloads
git --version
```

### **3.3 CUDA Toolkit** (if using GPU)
- **Windows**: https://developer.nvidia.com/cuda-downloads
- **Linux**: `sudo apt-get install nvidia-cuda-toolkit`
- **Check**: `nvidia-smi` (should show GPU info)

### **3.4 Ollama** (if running LLMs locally)
```bash
# Download from: https://ollama.ai/download
# After installation:
ollama pull llama3.3:70b-instruct-q4_K_M
# OR lighter version:
ollama pull llama3:8b-instruct
```

---

## 🔧 STEP 4: Project Setup Steps

### **4.1 Clone/Setup Repository**
```bash
# Navigate to your project directory
cd "c:\Users\krish\OneDrive - TUNI.fi\Desktop\Finland\RAG Agent"

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Windows CMD:
.\venv\Scripts\activate.bat
# Linux/Mac:
source venv/bin/activate
```

### **4.2 Install Dependencies**
```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Install additional system dependencies for spaCy
python -m spacy download en_core_web_sm
```

### **4.3 Download NLTK Data**
```python
# Run this Python script once:
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

---

## 🔐 STEP 5: Configuration Setup

### **5.1 Create Environment File**
Create a `.env` file in the project root (copy from `configs/config.yaml`):

```bash
# Copy config template
cp configs/config.yaml .env
```

### **5.2 Configure Models**

**For FARMI/CSC (Ollama):**
```yaml
LLM_PROVIDER: "ollama"
LLM_MODEL: "llama3.3:70b-instruct-q4_K_M"
OLLAMA_BASE_URL: "http://[FARMI_ENDPOINT]:11434"  # Get from supervisor
```

**For Local Ollama:**
```yaml
LLM_PROVIDER: "ollama"
LLM_MODEL: "llama3:8b-instruct"  # Lighter version
OLLAMA_BASE_URL: "http://localhost:11434"
```

**For OpenAI (if approved):**
```yaml
LLM_PROVIDER: "openai"
LLM_MODEL: "gpt-4"  # or "gpt-3.5-turbo"
# Add OPENAI_API_KEY to .env file
```

### **5.3 Hugging Face Setup**
```bash
# Install Hugging Face CLI
pip install huggingface_hub

# Login to Hugging Face (if models require authentication)
huggingface-cli login
# Enter your token from: https://huggingface.co/settings/tokens
```

### **5.4 Test Model Downloads**
```python
# Test script to verify models can be downloaded
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

# Test NLI model
print("Downloading NLI model...")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")
print("✓ NLI model ready")

# Test embedding model
print("Downloading embedding model...")
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("✓ Embedding model ready")
```

---

## 📊 STEP 6: Dataset Preparation

### **6.1 Required Datasets**

#### **FEVER Dataset** (Fact Extraction and VERification)
- **Purpose**: Factual verification evaluation
- **Download**: https://fever.ai/dataset/fever.html
- **Size**: ~1.5GB
- **Request**: Access/download permission

#### **TruthfulQA Dataset**
- **Purpose**: Misinformation detection evaluation
- **Download**: https://github.com/sylinrl/TruthfulQA
- **Size**: ~50MB
- **Request**: Access/download permission

#### **Poisoned Dataset** (To be created)
- **Purpose**: Knowledge poisoning attack evaluation
- **Action**: You'll create this during experiments
- **Storage**: `data/poisoned/`

### **6.2 Dataset Download Script**
```bash
# Create data directories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/poisoned

# Download FEVER (example - adjust based on actual download method)
# wget https://fever.ai/download/fever/wiki-pages.zip -P data/raw/
# unzip data/raw/wiki-pages.zip -d data/raw/
```

---

## 🧪 STEP 7: Testing Setup

### **7.1 Verify Installation**
```bash
# Test Python packages
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import langchain; print(f'LangChain: {langchain.__version__}')"
python -c "import faiss; print('FAISS: OK')"

# Test GPU (if available)
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

### **7.2 Test Model Access**
```python
# test_models.py
from transformers import pipeline
from sentence_transformers import SentenceTransformer

# Test NLI
print("Testing NLI model...")
nli = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
result = nli("This is a test", candidate_labels=["positive", "negative"])
print(f"✓ NLI working: {result}")

# Test Embeddings
print("Testing embedding model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode("Test sentence")
print(f"✓ Embeddings working: Shape {embeddings.shape}")
```

---

## 📝 STEP 8: Pre-Implementation Checklist

### **Before Starting Code Implementation:**

- [ ] **Infrastructure**
  - [ ] Python 3.10+ installed
  - [ ] Virtual environment created and activated
  - [ ] All dependencies installed (`pip install -r requirements.txt`)
  - [ ] GPU access confirmed (if needed)

- [ ] **Model Access**
  - [ ] LLM access confirmed (FARMI/Ollama/OpenAI)
  - [ ] Hugging Face account created and logged in
  - [ ] NLI model downloaded and tested
  - [ ] Embedding model downloaded and tested

- [ ] **Configuration**
  - [ ] `.env` file created from `configs/config.yaml`
  - [ ] Model endpoints configured
  - [ ] API keys added (if using OpenAI)
  - [ ] FARMI endpoint URL obtained (if applicable)

- [ ] **Data**
  - [ ] FEVER dataset access requested
  - [ ] TruthfulQA dataset access requested
  - [ ] Data directories created (`data/raw`, `data/processed`, `data/poisoned`)

- [ ] **Testing**
  - [ ] All imports working
  - [ ] Models loading successfully
  - [ ] Basic inference tests passing

---

## 🚨 Common Issues & Solutions

### **Issue 1: CUDA/GPU Not Available**
- **Solution**: Use CPU versions (`faiss-cpu` instead of `faiss-gpu`)
- **Impact**: Slower but functional

### **Issue 2: Model Download Fails**
- **Solution**: 
  - Check internet connection
  - Use Hugging Face CLI: `huggingface-cli download MODEL_NAME`
  - Set `HF_HOME` environment variable for custom cache location

### **Issue 3: Out of Memory**
- **Solution**: 
  - Use smaller models (8B instead of 70B)
  - Reduce batch size in config
  - Use quantization (Q4, Q8)

### **Issue 4: FARMI Access Issues**
- **Solution**: 
  - Contact supervisor for endpoint URL
  - Check VPN connection (if required)
  - Verify Ollama service is running on cluster

---

## 📞 Next Steps After Preparation

Once all items are checked:

1. **Verify Setup**: Run all test scripts
2. **Document Issues**: Note any problems encountered
3. **Contact Supervisor**: If any blockers remain
4. **Start Implementation**: Begin with retriever module

---

## 📚 Additional Resources

- **LangChain Docs**: https://python.langchain.com/
- **Hugging Face Docs**: https://huggingface.co/docs
- **FAISS Guide**: https://github.com/facebookresearch/faiss/wiki
- **Ollama Docs**: https://ollama.ai/docs
- **FARMI/CSC**: Contact supervisor for specific documentation

---

## ✅ Quick Start Command Sequence

```bash
# 1. Setup environment
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
pip install --upgrade pip
pip install -r requirements.txt

# 2. Download NLP models
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# 3. Test installation
python -c "import torch, transformers, langchain; print('All packages OK')"

# 4. Configure .env file
copy configs\config.yaml .env
# Edit .env with your settings

# 5. Test model access
python test_models.py  # (create this script)
```

---

**Last Updated**: February 2026  
**Project**: Trustworthy RAG - Master's Thesis  
**Author**: Balkrishna Giri
