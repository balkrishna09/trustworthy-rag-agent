# Deep Dive: Step 2 - Package Installation & Their Role in RAG Agent

## Overview

This document explains each package installed in Step 2, their functionality, and how they connect to your Trustworthy RAG Agent architecture.

---

## 🏗️ Architecture Context

Your RAG Agent follows this flow:
```
User Query → Retriever → [Evaluation Agent] → Generator → Response
                              ↓
                    ┌─────────────────────┐
                    │  NLI Verifier       │
                    │  Poison Detector    │
                    │  Trust Index Calc   │
                    └─────────────────────┘
```

Now let's see how each package fits into this architecture.

---

## 📦 Package Categories & Detailed Explanation

### 1. **LangChain Ecosystem** (RAG Framework Core)

#### `langchain` (>=0.1.0)
**What it does:**
- Core framework for building RAG (Retrieval-Augmented Generation) systems
- Provides abstractions for chains, agents, and memory
- Handles prompt templates, output parsers, and document loaders

**Role in your agent:**
- **Pipeline Integration**: Connects retriever → evaluation agent → generator
- **Document Processing**: Loads and processes documents for your knowledge base
- **Chain Building**: Creates the end-to-end RAG pipeline (`src/pipeline/`)
- **Memory Management**: Manages conversation context if needed

**Where used:**
- `src/pipeline/` - Main RAG pipeline
- `src/retriever/` - Document retrieval chains

---

#### `langchain-community` (>=0.1.0)
**What it does:**
- Community-contributed integrations
- Connectors to various vector stores, LLMs, and tools
- Extends LangChain with additional functionality

**Role in your agent:**
- **Vector Store Integration**: Connects to FAISS for document storage
- **LLM Providers**: Integrates with Ollama, OpenAI, HuggingFace
- **Document Loaders**: Loads documents from various sources

**Where used:**
- `src/retriever/` - FAISS vector store integration
- `src/generator/` - LLM provider connections

---

#### `langchain-huggingface` (>=0.0.3)
**What it does:**
- Specific integration for Hugging Face models
- Allows using Hugging Face models directly in LangChain chains
- Provides embeddings and LLM wrappers

**Role in your agent:**
- **Embedding Models**: Uses Hugging Face embedding models (`sentence-transformers/all-MiniLM-L6-v2`)
- **NLI Models**: Integrates NLI models for factual verification
- **Model Loading**: Simplifies loading Hugging Face models

**Where used:**
- `src/retriever/` - Embedding generation
- `src/evaluation_agent/` - NLI verification

---

### 2. **Vector Store & Embeddings** (Document Retrieval)

#### `faiss-cpu` (>=1.7.4)
**What it does:**
- Facebook AI Similarity Search - efficient vector similarity search
- Stores document embeddings in a searchable index
- Performs fast nearest neighbor search

**Role in your agent:**
- **Document Storage**: Stores all document embeddings (`data/processed/faiss_index`)
- **Similarity Search**: Finds most relevant documents for a query (TOP_K_RETRIEVAL: 5)
- **Retrieval Speed**: Enables fast document retrieval (critical for RAG)

**Where used:**
- `src/retriever/` - Core retrieval component
- Stores embeddings of documents in your knowledge base
- When user queries, searches for similar document chunks

**How it works:**
1. Documents are split into chunks (CHUNK_SIZE: 512)
2. Each chunk is converted to embedding vector
3. Vectors stored in FAISS index
4. Query embedding compared to all stored vectors
5. Returns top 5 most similar documents

---

#### `sentence-transformers` (>=2.2.2)
**What it does:**
- Converts text to dense vector embeddings
- Pre-trained models for semantic similarity
- Generates embeddings that capture semantic meaning

**Role in your agent:**
- **Query Embedding**: Converts user query to vector
- **Document Embedding**: Converts document chunks to vectors
- **Semantic Matching**: Enables semantic similarity search (not just keyword matching)

**Where used:**
- `src/retriever/` - Embedding generation
- Uses model: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional vectors)

**How it works:**
- Input: "What is the capital of France?"
- Output: [0.23, -0.45, 0.67, ...] (384 numbers representing meaning)
- Documents with similar embeddings are semantically related

---

### 3. **Hugging Face Ecosystem** (Models & Datasets)

#### `transformers` (>=4.36.0)
**What it does:**
- Library for using pre-trained transformer models
- Loads models like BERT, BART, GPT, etc.
- Provides model inference and fine-tuning capabilities

**Role in your agent:**
- **NLI Model**: Loads `facebook/bart-large-mnli` for factual verification
- **Text Classification**: Used by evaluation agent to verify facts
- **Model Inference**: Runs models for verification tasks

**Where used:**
- `src/evaluation_agent/` - NLI verifier component
- Verifies if retrieved documents support/refute the query

**How it works:**
- Takes query + retrieved document as input
- Outputs: SUPPORTS, REFUTES, or NEUTRAL
- This is your **factual verification** mechanism

---

#### `accelerate` (>=0.25.0)
**What it does:**
- Optimizes model inference speed
- Handles GPU/CPU distribution
- Manages model loading and execution

**Role in your agent:**
- **Performance**: Speeds up NLI model inference
- **Resource Management**: Efficiently uses CPU/GPU
- **Model Loading**: Optimizes loading of large models

**Where used:**
- `src/evaluation_agent/` - Behind the scenes for NLI model
- Makes verification faster

---

#### `datasets` (>=2.14.0)
**What it does:**
- Loads and processes datasets from Hugging Face
- Provides efficient data loading and preprocessing
- Handles large datasets with streaming

**Role in your agent:**
- **Dataset Loading**: Downloads FEVER, TruthfulQA datasets
- **Data Processing**: Prepares datasets for evaluation
- **Experiment Data**: Loads test datasets for evaluation

**Where used:**
- `download_datasets.py` - Dataset downloading
- `experiments/` - Loading evaluation datasets
- Testing your agent's performance

---

#### `huggingface-hub` (>=0.19.0)
**What it does:**
- Interface to Hugging Face Hub
- Downloads models and datasets
- Manages model caching

**Role in your agent:**
- **Model Download**: Downloads NLI model, embedding model
- **Dataset Download**: Downloads evaluation datasets
- **Caching**: Stores models locally for reuse

**Where used:**
- Model loading throughout the project
- Dataset downloading scripts

---

### 4. **Deep Learning Framework**

#### `torch` (>=2.1.0) - PyTorch
**What it does:**
- Deep learning framework
- Provides tensor operations and neural network building blocks
- Required by transformers and other ML libraries

**Role in your agent:**
- **Foundation**: Underlying framework for all ML models
- **Model Execution**: Runs NLI model, embedding model
- **Tensor Operations**: Handles all numerical computations

**Where used:**
- Everywhere models are used (embedding, NLI, etc.)
- Required dependency for transformers, sentence-transformers

**Note**: You have `torch 2.10.0+cpu` (CPU version) - no GPU needed for now

---

### 5. **NLI & Evaluation** (Core to Your Research)

#### `evaluate` (>=0.4.1)
**What it does:**
- Hugging Face evaluation metrics library
- Provides standard evaluation metrics
- Used for benchmarking model performance

**Role in your agent:**
- **Performance Metrics**: Evaluates your agent's accuracy
- **Benchmarking**: Compares against baseline systems (RQ3)
- **Experiments**: Measures detection rates for misinformation/poisoning

**Where used:**
- `experiments/` - Evaluation scripts
- Measuring how well your evaluation agent detects issues

---

#### `nltk` (>=3.8.1) - Natural Language Toolkit
**What it does:**
- Text processing and NLP utilities
- Tokenization, stemming, POS tagging
- Text preprocessing tools

**Role in your agent:**
- **Text Preprocessing**: Cleans and processes text before embedding
- **Tokenization**: Splits text into words/tokens
- **Text Analysis**: Analyzes text for poisoning detection

**Where used:**
- `src/retriever/` - Document preprocessing
- `src/evaluation_agent/` - Text analysis for poison detection

**Data downloaded:**
- punkt (tokenizer)
- stopwords (common words to filter)
- wordnet (semantic relationships)

---

#### `spacy` (>=3.7.2)
**What it does:**
- Advanced NLP library
- Named entity recognition, dependency parsing
- More sophisticated than NLTK

**Role in your agent:**
- **Text Analysis**: Advanced text processing
- **Entity Recognition**: Identifies entities in text
- **Dependency Parsing**: Understands sentence structure

**Where used:**
- `src/evaluation_agent/` - Advanced text analysis
- `src/retriever/` - Document processing

**Model downloaded:**
- `en_core_web_sm` - English language model

---

### 6. **Data Processing**

#### `pandas` (>=2.0.0)
**What it does:**
- Data manipulation and analysis
- Works with structured data (tables, CSV, JSON)
- Data cleaning and transformation

**Role in your agent:**
- **Dataset Handling**: Processes FEVER, TruthfulQA datasets
- **Results Analysis**: Analyzes experiment results
- **Data Preparation**: Prepares data for evaluation

**Where used:**
- `experiments/` - Processing evaluation results
- Dataset preprocessing

---

#### `numpy` (>=1.24.0)
**What it does:**
- Numerical computing library
- Array operations and mathematical functions
- Foundation for many ML libraries

**Role in your agent:**
- **Vector Operations**: Handles embedding vectors
- **Mathematical Operations**: Calculations for trust index
- **Array Processing**: Required by pandas, torch, etc.

**Where used:**
- Everywhere numerical operations are needed
- Trust index calculations
- Vector similarity computations

---

#### `tqdm` (>=4.66.0)
**What it does:**
- Progress bars for loops
- Shows progress during long operations

**Role in your agent:**
- **User Feedback**: Shows progress during:
  - Dataset downloading
  - Model loading
  - Batch processing
  - Experiment runs

**Where used:**
- Throughout the codebase for user-friendly progress indication

---

### 7. **Experiment Tracking**

#### `mlflow` (>=2.9.0)
**What it does:**
- Machine learning experiment tracking
- Logs parameters, metrics, models
- Tracks experiment history

**Role in your agent:**
- **Experiment Logging**: Tracks all your experiments
- **Metric Tracking**: Logs trust index scores, detection rates
- **Reproducibility**: Saves experiment configurations
- **Results Comparison**: Compares different approaches (RQ1, RQ2, RQ3)

**Where used:**
- `experiments/` - All experiment scripts
- Tracks which strategies work best for detection (RQ1)
- Logs trust index formulations (RQ2)
- Compares agent vs baseline (RQ3)

**Storage:**
- `mlruns/` directory (configured in config.yaml)

---

### 8. **Visualization**

#### `matplotlib` (>=3.8.0) & `seaborn` (>=0.13.0)
**What it does:**
- Data visualization libraries
- Creates charts, graphs, plots

**Role in your agent:**
- **Results Visualization**: Plots experiment results
- **Performance Charts**: Shows detection rates, trust scores
- **Thesis Figures**: Creates figures for your thesis

**Where used:**
- `experiments/` - Visualizing results
- `notebooks/` - Exploratory data analysis
- Creating figures for research questions

---

### 9. **Testing**

#### `pytest` (>=7.4.0) & `pytest-cov` (>=4.1.0)
**What it does:**
- Testing framework for Python
- Runs unit tests and integration tests
- Measures code coverage

**Role in your agent:**
- **Code Quality**: Ensures your code works correctly
- **Unit Tests**: Tests individual components
- **Integration Tests**: Tests full pipeline
- **Coverage**: Ensures all code is tested

**Where used:**
- `tests/` directory
- Testing retriever, evaluation agent, generator, pipeline

---

### 10. **Utilities**

#### `python-dotenv` (>=1.0.0)
**What it does:**
- Loads environment variables from `.env` file
- Manages configuration and secrets

**Role in your agent:**
- **Configuration**: Loads settings from `.env` file
- **API Keys**: Securely manages API keys (OpenAI, etc.)
- **Settings**: Loads model paths, endpoints, thresholds

**Where used:**
- Throughout the codebase
- Loads `configs/config.yaml` settings

---

#### `pyyaml` (>=6.0.1)
**What it does:**
- Parses YAML configuration files
- Reads structured configuration

**Role in your agent:**
- **Config Loading**: Reads `configs/config.yaml`
- **Settings Management**: Loads all configuration parameters
- **Trust Index Weights**: Loads TRUST_ALPHA, TRUST_BETA, TRUST_GAMMA

**Where used:**
- Configuration loading throughout the project

---

#### `loguru` (>=0.7.2)
**What it does:**
- Advanced logging library
- Better than standard Python logging
- Colored output, file rotation

**Role in your agent:**
- **Debugging**: Logs what's happening in your agent
- **Error Tracking**: Captures errors and warnings
- **Monitoring**: Tracks agent behavior during experiments

**Where used:**
- Throughout all modules
- Logs retrieval, evaluation, generation steps

---

### 11. **API Access** (LLM Integration)

#### `openai` (>=1.6.0)
**What it does:**
- OpenAI API client
- Accesses GPT models (GPT-4, GPT-3.5)
- Alternative to Ollama

**Role in your agent:**
- **LLM Provider**: Option for response generation
- **Backup Option**: If Ollama/FARMI not available
- **Generator**: Used in `src/generator/` module

**Where used:**
- `src/generator/` - If LLM_PROVIDER is "openai"
- Alternative to Ollama for generating responses

---

#### `ollama` (>=0.1.0)
**What it does:**
- Ollama API client
- Accesses local/remote Ollama models
- Connects to FARMI/CSC cluster

**Role in your agent:**
- **Primary LLM Provider**: Main way to access LLM (via FARMI)
- **Response Generation**: Generates final responses
- **Model Access**: Connects to `llama3.3:70b-instruct-q4_K_M`

**Where used:**
- `src/generator/` - Primary LLM access
- Connects to FARMI endpoint (OLLAMA_BASE_URL)
- Generates responses after evaluation

---

## 🔗 How Packages Connect in Your Architecture

### **Retrieval Flow:**
```
User Query
    ↓
sentence-transformers → Query Embedding (384-dim vector)
    ↓
faiss-cpu → Search similar documents (TOP_K=5)
    ↓
langchain → Format retrieved documents
```

### **Evaluation Flow:**
```
Retrieved Documents + Query
    ↓
transformers → Load NLI model (bart-large-mnli)
    ↓
nltk/spacy → Preprocess text
    ↓
evaluate → Calculate metrics
    ↓
numpy → Calculate Trust Index (alpha, beta, gamma weights)
    ↓
loguru → Log results
```

### **Generation Flow:**
```
Evaluated Documents + Query
    ↓
ollama/openai → Connect to LLM
    ↓
langchain → Format prompt
    ↓
torch → Model inference (if using HuggingFace models)
    ↓
Response Generated
```

### **Experiment Flow:**
```
datasets → Load FEVER/TruthfulQA
    ↓
pandas → Process data
    ↓
Run experiments
    ↓
mlflow → Log parameters & metrics
    ↓
matplotlib/seaborn → Visualize results
```

---

## 📊 Package Dependencies Map

```
torch (Foundation)
    ├── transformers (NLI models)
    ├── sentence-transformers (Embeddings)
    └── accelerate (Optimization)

langchain (RAG Framework)
    ├── langchain-community (Integrations)
    ├── langchain-huggingface (HF models)
    ├── faiss-cpu (Vector store)
    └── ollama/openai (LLM providers)

evaluation_agent/
    ├── transformers (NLI verification)
    ├── nltk/spacy (Text processing)
    ├── numpy (Trust index calculation)
    └── evaluate (Metrics)

experiments/
    ├── datasets (Load test data)
    ├── pandas (Process results)
    ├── mlflow (Track experiments)
    └── matplotlib/seaborn (Visualize)
```

---

## 🎯 Summary: What Each Component Does

| Component | Package | Purpose |
|-----------|---------|---------|
| **Retriever** | sentence-transformers, faiss-cpu, langchain | Finds relevant documents |
| **NLI Verifier** | transformers, nltk, spacy | Verifies factual consistency |
| **Poison Detector** | transformers, nltk, numpy | Detects adversarial content |
| **Trust Index** | numpy, pyyaml | Calculates reliability score |
| **Generator** | ollama/openai, langchain | Generates final response |
| **Pipeline** | langchain, loguru | Connects all components |
| **Experiments** | datasets, mlflow, pandas | Evaluates performance |

---

This completes the deep dive into Step 2! Each package has a specific role in making your Trustworthy RAG Agent work.
