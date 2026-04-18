# Model Access Request Template

**Subject**: Model Access Request for Trustworthy RAG Thesis Project

---

Dear [Supervisor Name],

I am working on my Master's thesis project "Trustworthy RAG: An Evaluation Agent for Detecting Misinformation and Knowledge Poisoning in Generative AI Systems" and need access to the following resources.

## 1. Large Language Model (LLM) Access

**Required for**: Response generation in the RAG pipeline

**Options** (please indicate which is available):

### Option A: FARMI/CSC Cluster Access (Preferred)
- **Service**: Ollama service on FARMI/CSC cluster
- **Model**: `llama3.3:70b-instruct-q4_K_M` or similar
- **What I need**:
  - Access credentials/account
  - Endpoint URL for Ollama service
  - Usage guidelines/quota information

### Option B: GPU Compute Resources
- **Requirements**: 
  - Minimum: 16GB VRAM GPU
  - Recommended: 32GB+ VRAM for 70B models
  - Alternative: Access to smaller models (8B-13B) with less VRAM
- **What I need**:
  - GPU node access
  - SSH credentials
  - Installation guide for Ollama on cluster

### Option C: OpenAI API Access
- **Model**: GPT-4 or GPT-3.5-turbo
- **What I need**:
  - API key
  - Budget/quota information
  - Usage limits

**Priority**: Option A (FARMI) > Option B (GPU) > Option C (OpenAI)

---

## 2. Hugging Face Model Access

**Required for**: 
- Natural Language Inference (NLI) for factual verification
- Text embeddings for document retrieval

### Models Needed:

1. **NLI Model**: `facebook/bart-large-mnli`
   - **Size**: ~1.6GB
   - **Purpose**: Verify factual consistency between query and retrieved documents
   - **Alternative**: `microsoft/deberta-large-mnli` (~1.3GB)

2. **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
   - **Size**: ~80MB
   - **Purpose**: Generate embeddings for document retrieval
   - **Alternative**: `BAAI/bge-m3` (~420MB)

**What I need**:
- Hugging Face account access (or instructions to create one)
- Download permissions (if restricted)
- Storage location: ~3GB total

**Note**: These models are typically publicly available, but I want to confirm access and any download restrictions.

---

## 3. Computing Resources

### Minimum Requirements:
- **RAM**: 16GB
- **Storage**: 50GB (for datasets, models, experiments)
- **CPU**: 4 cores
- **GPU**: Optional but recommended

### Recommended:
- **RAM**: 32GB+
- **Storage**: 100GB+
- **CPU**: 8+ cores
- **GPU**: 16GB+ VRAM (if running models locally)

**What I need**:
- Confirmation of available resources
- Access method (local machine, cluster, cloud)
- Any usage quotas or limitations

---

## 4. Dataset Access

**Required datasets**:

1. **FEVER Dataset** (Fact Extraction and VERification)
   - **Source**: https://fever.ai/dataset/fever.html
   - **Size**: ~1.5GB
   - **Purpose**: Evaluate factual verification capabilities

2. **TruthfulQA Dataset**
   - **Source**: https://github.com/sylinrl/TruthfulQA
   - **Size**: ~50MB
   - **Purpose**: Evaluate misinformation detection

**What I need**:
- Download permissions/access
- Storage location confirmation
- Any preprocessing requirements

---

## 5. Additional Services

### MLflow (Experiment Tracking)
- **Status**: Already in requirements.txt
- **Question**: Should I use local MLflow or is there a shared instance?

### GitHub/GitLab Access
- **Question**: Is there a university Git repository I should use?

---

## Timeline

- **Immediate**: Need model access to start implementation
- **Project Duration**: [Your timeline]
- **Expected Usage**: 
  - Development: [X hours/week]
  - Experiments: [Y hours/week]

---

## Questions

1. Which LLM access option is available/preferred?
2. Are there any restrictions on model downloads from Hugging Face?
3. What is the preferred storage location for models and datasets?
4. Are there any compute quotas or time limits I should be aware of?
5. Is there documentation for accessing FARMI/CSC resources?

---

Thank you for your support. Please let me know which resources are available and how to proceed.

Best regards,  
[Your Name]  
[Your Email]  
[Your Student ID]

---

**Project Details**:
- **Title**: Trustworthy RAG: An Evaluation Agent for Detecting Misinformation and Knowledge Poisoning in Generative AI Systems
- **Institution**: Tampere University
- **Program**: Master's in Information Security
- **Year**: 2026
