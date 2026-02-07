# Quick Start Guide

## 🚀 Fastest Path to Get Started

Follow these steps in order to prepare your environment:

### Step 1: Initial Setup (5 minutes)

```powershell
# Navigate to project directory
cd "c:\Users\krish\OneDrive - TUNI.fi\Desktop\Finland\RAG Agent"

# Create virtual environment
python -m venv venv

# Activate virtual environment (PowerShell)
.\venv\Scripts\Activate.ps1

# If activation fails, run this first:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Step 2: Automated Setup (10-15 minutes)

```powershell
# Run automated setup script
python setup_environment.py
```

This will:
- ✓ Install all Python packages
- ✓ Download NLTK data
- ✓ Download spaCy models
- ✓ Create directory structure
- ✓ Create .env configuration file

### Step 3: Verify Setup (2 minutes)

```powershell
# Run verification script
python test_setup.py
```

Check that all critical items show ✓ PASS

### Step 4: Request Resources from Supervisor (1 day)

1. Open `SUPERVISOR_REQUEST_TEMPLATE.md`
2. Fill in your details
3. Send to supervisor
4. Wait for access confirmation

**What you need:**
- LLM access (FARMI/Ollama preferred)
- Hugging Face model access
- Dataset access (FEVER, TruthfulQA)

### Step 5: Configure Environment (5 minutes)

```powershell
# Edit .env file with your settings
notepad .env
```

**Key settings to update:**
- `OLLAMA_BASE_URL`: Get from supervisor (if using FARMI)
- `LLM_MODEL`: Confirm which model is available
- Add `OPENAI_API_KEY`: If using OpenAI instead

### Step 6: Test Model Access (5 minutes)

```powershell
# Test if models can be downloaded
python -c "from transformers import pipeline; nli = pipeline('zero-shot-classification', model='facebook/bart-large-mnli'); print('NLI Model OK')"
python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); print('Embedding Model OK')"
```

### Step 7: Download Datasets (30 minutes - 1 hour)

Once you have access:
- Download FEVER dataset to `data/raw/`
- Download TruthfulQA dataset to `data/raw/`

---

## ⚡ Troubleshooting

### Virtual Environment Won't Activate

**PowerShell:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\venv\Scripts\Activate.ps1
```

**CMD:**
```cmd
.\venv\Scripts\activate.bat
```

### Package Installation Fails

```powershell
# Upgrade pip first
python -m pip install --upgrade pip

# Try installing individually
pip install torch transformers langchain
```

### Models Won't Download

```powershell
# Login to Hugging Face
huggingface-cli login

# Or set cache directory
set HF_HOME=C:\huggingface_cache
```

### CUDA/GPU Issues

If you don't have GPU:
- Use `faiss-cpu` (already in requirements.txt)
- Models will run on CPU (slower but works)

---

## 📋 Checklist

Before starting implementation:

- [ ] Virtual environment created and activated
- [ ] All packages installed (`pip install -r requirements.txt`)
- [ ] NLTK data downloaded
- [ ] spaCy model downloaded
- [ ] `.env` file created and configured
- [ ] Supervisor request sent
- [ ] LLM access confirmed
- [ ] Hugging Face models tested
- [ ] Datasets downloaded
- [ ] `test_setup.py` shows all critical checks passing

---

## 🎯 What's Next?

Once everything is checked:

1. **Read**: `PREPARATION_GUIDE.md` for detailed information
2. **Review**: `README.md` for project architecture
3. **Start**: Begin implementation with retriever module

---

## 📞 Need Help?

- **Setup Issues**: Check `PREPARATION_GUIDE.md` → Common Issues section
- **Model Access**: See `SUPERVISOR_REQUEST_TEMPLATE.md`
- **Project Structure**: See `README.md`

---

**Estimated Total Setup Time**: 1-2 hours (excluding supervisor response time)
