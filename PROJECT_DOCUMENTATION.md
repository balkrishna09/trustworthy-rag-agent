# Trustworthy RAG Agent - Complete Project Documentation

**Author:** Balkrishna Giri  
**Project:** Master's Thesis - Tampere University  
**Last Updated:** February 2026  

---

## Table of Contents

1. [What is this Project?](#1-what-is-this-project)
2. [The Big Picture - How Everything Connects](#2-the-big-picture---how-everything-connects)
3. [Project Structure](#3-project-structure)
4. [Module 1: Retriever](#4-module-1-retriever)
5. [Module 2: Generator](#5-module-2-generator)
6. [Module 3: Pipeline](#6-module-3-pipeline)
7. [Module 4: Evaluation Agent](#7-module-4-evaluation-agent-coming-soon)
8. [Configuration](#8-configuration)
9. [How to Use](#9-how-to-use)
10. [Glossary](#10-glossary)

---

## 1. What is this Project?

### Simple Explanation

Imagine you have a **very smart robot assistant** that can answer your questions. But here's the problem: sometimes this robot might give you **wrong information** or even **lie** (either by mistake or because someone tricked it).

**Your project builds a "lie detector" for this robot.**

When the robot answers a question, your system will:
1. Check if the answer is **factually correct**
2. Check if someone tried to **trick the robot** with fake information
3. Give a **trust score** - how much you should trust this answer

### Technical Explanation

This is a **RAG (Retrieval-Augmented Generation)** system with an **Evaluation Agent**:

- **RAG**: Instead of the AI making up answers, it first searches through documents to find relevant information, then generates an answer based on what it found.
- **Evaluation Agent**: Checks if the retrieved documents and generated answer are trustworthy.

### Why is this Important?

- AI systems can be **fooled** by injecting false information into their knowledge base (called "knowledge poisoning")
- AI can **hallucinate** (make up facts that aren't true)
- We need ways to **detect** when AI is giving unreliable answers

---

## 2. The Big Picture - How Everything Connects

### The Flow (Like an Assembly Line)

```
USER ASKS A QUESTION
        ↓
┌───────────────────┐
│     RETRIEVER     │  ← "Finds relevant documents"
│                   │     (Like a librarian finding books)
└─────────┬─────────┘
          ↓
┌───────────────────┐
│ EVALUATION AGENT  │  ← "Checks if documents are trustworthy"
│                   │     (Like a fact-checker)
│  • NLI Verifier   │
│  • Poison Detector│
│  • Trust Index    │
└─────────┬─────────┘
          ↓
┌───────────────────┐
│     GENERATOR     │  ← "Creates the answer"
│                   │     (Like a writer using the books)
└─────────┬─────────┘
          ↓
    ANSWER + TRUST SCORE
```

### Real-World Analogy

Think of it like a **research assistant**:

1. **Retriever** = The assistant goes to the library and finds relevant books
2. **Evaluation Agent** = The assistant checks if the books are from reliable sources
3. **Generator** = The assistant writes a summary based on the books
4. **Trust Score** = The assistant tells you "I'm 85% confident this is correct"

---

## 3. Project Structure

```
RAG Agent/
│
├── configs/                    # Settings and configuration
│   └── config.yaml            # All settings in one place
│
├── data/                      # Datasets
│   ├── raw/                   # Original datasets
│   │   ├── fever/            # FEVER fact-checking dataset
│   │   └── truthfulqa/       # TruthfulQA dataset
│   ├── processed/            # Processed data
│   └── poisoned/             # Test data with fake info (for testing)
│
├── src/                       # Source code (THE MAIN CODE)
│   │
│   ├── retriever/            # Module 1: Finding documents
│   │   ├── __init__.py       # Makes this a Python package
│   │   ├── embeddings.py     # Converts text to numbers
│   │   ├── document_processor.py  # Loads and chunks documents
│   │   ├── vector_store.py   # Stores and searches documents
│   │   └── retriever.py      # Main retriever class
│   │
│   ├── generator/            # Module 2: Generating answers
│   │   ├── __init__.py       # Makes this a Python package
│   │   ├── llm_client.py     # Talks to the AI (FARMI)
│   │   ├── prompts.py        # Templates for questions
│   │   └── generator.py      # Main generator class
│   │
│   ├── pipeline/             # Module 3: Connects everything
│   │   ├── __init__.py       # Makes this a Python package
│   │   └── rag_pipeline.py   # The complete system
│   │
│   └── evaluation_agent/     # Module 4: Checks trustworthiness
│       └── __init__.py       # (Coming soon!)
│
└── Various documentation files
```

---

## 4. Module 1: Retriever

### What is a Retriever?

**Simple:** A retriever is like a **search engine** for your documents. When you ask a question, it finds the most relevant documents that might contain the answer.

**Example:**
- You ask: "What is the capital of France?"
- Retriever finds: "Paris is the capital of France. The Eiffel Tower is in Paris."

### Why do we need it?

Without a retriever, the AI would have to **memorize everything** or **make up answers**. With a retriever:
- The AI can search through documents
- Answers are based on **actual information**
- We can update the knowledge by adding new documents

### Files in the Retriever Module

#### File 1: `embeddings.py` - The Translator

**What it does:** Converts text into numbers (called "embeddings" or "vectors").

**Why numbers?** Computers can't understand text directly. By converting text to numbers, we can compare how similar two pieces of text are.

**Simple analogy:** It's like converting colors to RGB numbers. "Red" becomes (255, 0, 0). Now computers can compare colors mathematically.

**Code explanation:**

```python
# This is what the EmbeddingGenerator does:

class EmbeddingGenerator:
    def __init__(self):
        # Load a pre-trained model that knows how to convert text to numbers
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def embed_query(self, text):
        # Convert text to a list of 384 numbers
        # "Paris is beautiful" → [0.23, -0.45, 0.67, ...]
        return self.model.encode(text)
```

**Key points:**
- Uses `sentence-transformers` library
- Outputs 384 numbers for each piece of text
- Similar texts will have similar numbers

---

#### File 2: `document_processor.py` - The Organizer

**What it does:** 
1. Loads documents from files
2. Splits long documents into smaller chunks

**Why split documents?**
- AI models have a limit on how much text they can process
- Smaller chunks = more precise retrieval
- If a document is 10 pages, we split it into paragraphs

**Simple analogy:** It's like cutting a pizza into slices. Easier to handle and serve!

**Code explanation:**

```python
# Document class - holds one piece of text
class Document:
    def __init__(self, content, metadata):
        self.content = "Paris is the capital..."  # The actual text
        self.metadata = {"source": "geography.txt"}  # Where it came from

# TextChunker - splits long text
class TextChunker:
    def __init__(self, chunk_size=512):
        self.chunk_size = 512  # Max characters per chunk
    
    def split_text(self, text):
        # Split long text into smaller pieces
        # "Very long document..." → ["Chunk 1...", "Chunk 2...", "Chunk 3..."]
        chunks = []
        # ... splitting logic ...
        return chunks

# DocumentLoader - loads files
class DocumentLoader:
    def load_text_file(self, path):
        # Read a file and return a Document
        with open(path) as f:
            content = f.read()
        return Document(content=content)
    
    def load_fever_dataset(self, path):
        # Load FEVER dataset (special format)
        # Returns list of claims with labels
        pass
```

**Key points:**
- `Document`: Holds text + metadata (like source file name)
- `TextChunker`: Splits long text into ~512 character pieces
- `DocumentLoader`: Knows how to load different file types

---

#### File 3: `vector_store.py` - The Library

**What it does:** Stores document embeddings and finds similar ones quickly.

**Simple analogy:** Imagine a library where books are organized by topic. When you want books about "France", you go to the France section. The vector store does this automatically using math!

**How it works:**
1. Takes document embeddings (numbers)
2. Stores them in a special structure (FAISS)
3. When you search, it finds the closest numbers

**Code explanation:**

```python
class FAISSVectorStore:
    def __init__(self, dimension=384):
        # Create an index to store vectors
        # dimension = how many numbers per embedding (384)
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []  # Store original documents too
    
    def add_documents(self, documents, embeddings):
        # Add documents and their embeddings to the store
        self.index.add(embeddings)  # Add numbers to FAISS
        self.documents.extend(documents)  # Keep track of original docs
    
    def search(self, query_embedding, k=5):
        # Find k most similar documents
        # query_embedding = [0.23, -0.45, ...] (the question as numbers)
        # Returns: [(Document1, score1), (Document2, score2), ...]
        distances, indices = self.index.search(query_embedding, k)
        results = []
        for idx, dist in zip(indices, distances):
            results.append((self.documents[idx], dist))
        return results
```

**Key points:**
- Uses FAISS library (Facebook AI Similarity Search)
- Very fast - can search millions of documents in milliseconds
- Returns documents sorted by similarity (best match first)

---

#### File 4: `retriever.py` - The Coordinator

**What it does:** Combines all the above pieces into one easy-to-use class.

**Simple analogy:** If the other files are workers (translator, organizer, librarian), this file is the **manager** who coordinates them all.

**Code explanation:**

```python
class Retriever:
    def __init__(self):
        # Create all the components
        self.embedder = EmbeddingGenerator()  # The translator
        self.chunker = TextChunker()           # The organizer
        self.vector_store = FAISSVectorStore() # The library
    
    def add_documents(self, documents):
        # Step 1: Split documents into chunks
        chunked_docs = self.chunker.split_documents(documents)
        
        # Step 2: Convert chunks to embeddings
        texts = [doc.content for doc in chunked_docs]
        embeddings = self.embedder.embed_documents(texts)
        
        # Step 3: Store in vector store
        self.vector_store.add_documents(chunked_docs, embeddings)
    
    def retrieve(self, query, k=5):
        # Step 1: Convert query to embedding
        query_embedding = self.embedder.embed_query(query)
        
        # Step 2: Search for similar documents
        results = self.vector_store.search(query_embedding, k)
        
        # Returns: [(Document1, score1), (Document2, score2), ...]
        return results
```

**How the retriever works (step by step):**

```
1. You call: retriever.add_documents(["Paris is capital of France", ...])
   
   Documents → Chunker → Embedder → Vector Store
   "Paris..."  →  "Paris..." →  [0.23, ...]  →  Stored!

2. You call: retriever.retrieve("capital of France?")
   
   Query → Embedder → Vector Store → Results
   "capital...?" →  [0.45, ...]  →  Search  →  "Paris is capital..."
```

---

## 5. Module 2: Generator

### What is a Generator?

**Simple:** The generator is the AI that **writes the answer** based on the documents found by the retriever.

**Example:**
- Retriever found: "Paris is the capital of France."
- Question: "What is the capital of France?"
- Generator writes: "The capital of France is Paris."

### Why do we need it?

The retriever only finds relevant documents. Someone needs to **read** those documents and **write** a proper answer. That's what the generator does using a Large Language Model (LLM).

### What is an LLM?

**LLM = Large Language Model** (like ChatGPT, Llama, etc.)

It's a very large AI trained on billions of texts that can:
- Understand language
- Write like a human
- Answer questions
- Summarize information

**We use:** `llama3.3:70b` on FARMI (university's AI server)

### Files in the Generator Module

#### File 1: `llm_client.py` - The Phone Line

**What it does:** Communicates with the LLM server (FARMI).

**Simple analogy:** Like making a phone call. You dial a number (API URL), say something (prompt), and get a response (answer).

**Code explanation:**

```python
class FARMIClient:
    def __init__(self, api_url, api_key, model):
        # Connection details
        self.api_url = "https://gptlab.rd.tuni.fi/students/ollama/v1"
        self.api_key = "sk-ollama-..."  # Your secret key
        self.model = "llama3.3:70b"      # Which AI to use
    
    def generate(self, prompt):
        # Send request to FARMI
        response = requests.post(
            url=self.api_url + "/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "prompt": prompt,
                "max_tokens": 512
            }
        )
        
        # Get the answer
        return response.json()["choices"][0]["text"]
```

**Key points:**
- Sends HTTP requests to FARMI server
- Uses your API key for authentication
- Returns the AI's response as text

---

#### File 2: `prompts.py` - The Script Writer

**What it does:** Creates the text (prompt) that we send to the AI.

**Why do we need prompts?** The AI doesn't know what you want it to do. You need to give clear instructions.

**Simple analogy:** Like giving instructions to a new employee. "Here are some documents. Read them and answer this question."

**Code explanation:**

```python
# A template for RAG questions
RAG_PROMPT = """
You are a helpful assistant that answers questions based on the provided context.
Use ONLY the information from the context below to answer the question.

Context:
{context}

Question: {question}

Answer:
"""

def create_rag_prompt(question, context_docs):
    # Combine documents into one context string
    context = "\n".join([
        f"[1] {context_docs[0]}",
        f"[2] {context_docs[1]}",
        # ...
    ])
    
    # Fill in the template
    prompt = RAG_PROMPT.format(
        context=context,
        question=question
    )
    
    return prompt

# Example output:
"""
You are a helpful assistant...

Context:
[1] Paris is the capital of France.
[2] The Eiffel Tower is in Paris.

Question: What is the capital of France?

Answer:
"""
```

**Key points:**
- Templates tell the AI how to behave
- We inject the retrieved documents as "context"
- The AI must answer based on the context (not make things up)

---

#### File 3: `generator.py` - The Writer

**What it does:** Combines the LLM client and prompts to generate answers.

**Code explanation:**

```python
class Generator:
    def __init__(self, config):
        # Create the phone line to FARMI
        self.llm_client = FARMIClient(
            api_url=config['FARMI_API_URL'],
            api_key=config['FARMI_API_KEY'],
            model=config['LLM_MODEL']
        )
    
    def generate(self, question, context_docs):
        # Step 1: Create the prompt
        prompt = create_rag_prompt(question, context_docs)
        
        # Step 2: Send to LLM and get response
        response = self.llm_client.generate(prompt)
        
        return response

# Usage:
generator = Generator(config)
answer = generator.generate(
    question="What is the capital of France?",
    context_docs=["Paris is the capital of France."]
)
# answer = "The capital of France is Paris."
```

---

## 6. Module 3: Pipeline

### What is a Pipeline?

**Simple:** The pipeline connects everything together. It's the **main program** that you actually use.

**Analogy:** If retriever, generator, and evaluation agent are workers, the pipeline is the **assembly line** that coordinates them.

### File: `rag_pipeline.py`

**What it does:**
1. Takes a user question
2. Calls the retriever to find documents
3. (Soon) Calls the evaluation agent to check trustworthiness
4. Calls the generator to write the answer
5. Returns the answer with trust score

**Code explanation:**

```python
class RAGPipeline:
    def __init__(self, config):
        # Create all components
        self.retriever = Retriever(config)
        self.generator = Generator(config)
        # self.evaluation_agent = EvaluationAgent(config)  # Coming soon!
    
    def query(self, question):
        # STEP 1: Find relevant documents
        retrieved = self.retriever.retrieve(question, k=5)
        # retrieved = [(Doc1, 0.95), (Doc2, 0.87), ...]
        
        # Extract just the text content
        docs = [doc.content for doc, score in retrieved]
        
        # STEP 2: (Coming soon) Evaluate documents
        # evaluation = self.evaluation_agent.evaluate(question, docs)
        
        # STEP 3: Generate answer
        response = self.generator.generate(question, docs)
        
        # STEP 4: Return everything
        return RAGResponse(
            query=question,
            response=response,
            retrieved_docs=docs,
            scores=[score for _, score in retrieved]
        )

# Usage:
pipeline = RAGPipeline(config)
pipeline.add_documents(["Paris is capital of France", "London is capital of UK"])
result = pipeline.query("What is the capital of France?")
print(result.response)  # "The capital of France is Paris."
```

### Visual Flow

```
pipeline.query("What is the capital of France?")
                    │
                    ▼
        ┌───────────────────┐
        │     RETRIEVER     │
        │                   │
        │ "Find documents   │
        │  about France"    │
        └─────────┬─────────┘
                  │
                  ▼
        Found: "Paris is the capital of France"
                  │
                  ▼
        ┌───────────────────┐
        │     GENERATOR     │
        │                   │
        │ "Write answer     │
        │  using documents" │
        └─────────┬─────────┘
                  │
                  ▼
        Answer: "The capital of France is Paris."
```

---

## 7. Module 4: Evaluation Agent (IMPLEMENTED)

### What it Does

This is the **core research component** of your thesis! The Evaluation Agent checks if the RAG system's answer is trustworthy by:

1. Verifying factual consistency (NLI Verifier)
2. Detecting potential poisoning (Poison Detector)
3. Calculating a composite trust score (Trust Index)

### Files in the Evaluation Agent Module

```
src/evaluation_agent/
├── __init__.py           # Module exports
├── nli_verifier.py       # Factual consistency checker
├── poison_detector.py    # Adversarial content detector
├── trust_index.py        # Trust score calculator
└── evaluation_agent.py   # Main orchestrator
```

---

### File 1: `nli_verifier.py` - The Fact Checker

**What it does:** Uses Natural Language Inference (NLI) to check if the generated answer is supported by the retrieved documents.

**Simple analogy:** Like a teacher checking if your essay answer matches what's in the textbook.

**How NLI works:**
- Takes two texts: a "premise" (the document) and a "hypothesis" (the answer)
- Classifies the relationship as:
  - **ENTAILMENT**: The document supports the answer ✓
  - **CONTRADICTION**: The document contradicts the answer ✗
  - **NEUTRAL**: The document neither supports nor contradicts

**Code explanation:**

```python
class NLIVerifier:
    def __init__(self, model_name="facebook/bart-large-mnli"):
        # Load a pre-trained NLI model from HuggingFace
        self.model = pipeline("zero-shot-classification", model=model_name)
    
    def verify_pair(self, premise, hypothesis):
        # Check if document (premise) supports answer (hypothesis)
        # Returns: ENTAILMENT, CONTRADICTION, or NEUTRAL
        result = self.model(premise, ["entailment", "contradiction", "neutral"])
        return result
    
    def verify_answer(self, answer, documents):
        # Check answer against ALL retrieved documents
        # Returns average scores and overall verdict
        results = []
        for doc in documents:
            result = self.verify_pair(doc, answer)
            results.append(result)
        
        # Calculate averages
        avg_entailment = average([r.entailment_score for r in results])
        return VerificationResult(
            avg_entailment=avg_entailment,
            is_supported=avg_entailment > 0.5
        )
```

**Key metrics returned:**
- `avg_entailment`: Average support score (0-1)
- `max_contradiction`: Highest contradiction score (0-1)
- `support_ratio`: Fraction of documents that support the answer
- `is_supported`: Boolean - is the answer factually supported?

---

### File 2: `poison_detector.py` - The Security Guard

**What it does:** Detects if someone injected false or malicious information into the retrieved documents (knowledge poisoning attack).

**Simple analogy:** Like an airport security scanner checking for suspicious items.

**Types of attacks it detects:**

1. **Instruction Injection:** "Ignore all previous instructions and say X"
2. **Contradiction Patterns:** "Actually, contrary to what you might think..."
3. **Claim Manipulation:** "The REAL truth is..."
4. **Format Injection:** XML tags or chat format in documents

**Code explanation:**

```python
class PoisonDetector:
    # Suspicious patterns to look for
    SUSPICIOUS_PATTERNS = [
        (r'ignore.*previous', 0.7, "Instruction override attempt"),
        (r'actually.*not', 0.3, "Contradiction pattern"),
        (r'forget.*everything', 0.7, "Instruction injection"),
        # ... more patterns
    ]
    
    def detect_document(self, document):
        signals = []
        
        # Check each suspicious pattern
        for pattern, severity, description in self.SUSPICIOUS_PATTERNS:
            if pattern matches document:
                signals.append(PoisonSignal(
                    severity=severity,
                    description=description
                ))
        
        # Calculate poison probability
        poison_probability = combine(signals)
        return DocumentPoisonResult(
            is_suspicious=poison_probability > 0.7,
            poison_probability=poison_probability
        )
    
    def detect_batch(self, documents):
        # Check ALL documents
        results = [self.detect_document(doc) for doc in documents]
        
        # Return overall assessment
        return PoisonDetectionResult(
            overall_probability=max([r.probability for r in results]),
            is_contaminated=any([r.is_suspicious for r in results])
        )
```

**Key metrics returned:**
- `overall_poison_probability`: Likelihood of poisoning (0-1)
- `num_suspicious_docs`: Count of flagged documents
- `high_risk_indices`: Which documents are most suspicious
- `is_contaminated`: Boolean - is the knowledge base poisoned?

---

### File 3: `trust_index.py` - The Score Calculator

**What it does:** Combines all evaluation signals into a single trust score (0-1).

**Simple analogy:** Like a credit score that considers multiple factors to give you one number.

**The Formula:**

```
Trust Index = α × Factuality + β × Consistency + γ × (1 - PoisonProbability)

Where:
- α = 0.4 (40% weight for factuality)
- β = 0.35 (35% weight for consistency)
- γ = 0.25 (25% weight for poison safety)
```

**Code explanation:**

```python
class TrustIndexCalculator:
    def __init__(self, alpha=0.4, beta=0.35, gamma=0.25):
        self.alpha = alpha  # Factuality weight
        self.beta = beta    # Consistency weight
        self.gamma = gamma  # Poison safety weight
    
    def calculate(self, factuality, consistency, poison_score):
        # Calculate weighted sum
        trust_score = (
            self.alpha * factuality +
            self.beta * consistency +
            self.gamma * (1 - poison_score)  # Invert poison score!
        )
        
        # Determine trust level
        if trust_score > 0.8:
            level = "HIGH"
        elif trust_score > 0.5:
            level = "MEDIUM"
        elif trust_score > 0.3:
            level = "LOW"
        else:
            level = "VERY_LOW"
        
        return TrustIndexResult(
            trust_score=trust_score,
            trust_level=level,
            is_trustworthy=trust_score >= 0.5
        )
```

**Trust Levels:**

| Score Range | Level | Meaning |
|-------------|-------|---------|
| > 0.8 | HIGH | Answer is reliable |
| 0.5 - 0.8 | MEDIUM | Answer is probably OK, verify if important |
| 0.3 - 0.5 | LOW | Answer may have issues, verify independently |
| < 0.3 | VERY_LOW | Do NOT trust this answer |

**Example calculation:**
- Factuality: 0.9 (answer matches documents well)
- Consistency: 0.85 (documents agree with each other)
- Poison Probability: 0.1 (low chance of poisoning)

```
Trust = 0.4×0.9 + 0.35×0.85 + 0.25×(1-0.1)
      = 0.36 + 0.30 + 0.23
      = 0.89 (HIGH trust)
```

---

### File 4: `evaluation_agent.py` - The Orchestrator

**What it does:** Combines all three components (NLI, Poison, Trust) into one easy-to-use class.

**Simple analogy:** Like a manager who coordinates the fact-checker, security guard, and score calculator.

**Code explanation:**

```python
class EvaluationAgent:
    def __init__(self, config):
        # Initialize all components
        self.nli_verifier = NLIVerifier(config['NLI_MODEL'])
        self.poison_detector = PoisonDetector()
        self.trust_calculator = TrustIndexCalculator(
            alpha=config['TRUST_ALPHA'],
            beta=config['TRUST_BETA'],
            gamma=config['TRUST_GAMMA']
        )
    
    def evaluate(self, query, response, documents):
        # Step 1: Check factual consistency
        nli_result = self.nli_verifier.verify_answer(response, documents)
        
        # Step 2: Check for poisoning
        poison_result = self.poison_detector.detect_batch(documents)
        
        # Step 3: Calculate trust index
        trust_result = self.trust_calculator.calculate(
            factuality=nli_result.avg_entailment,
            consistency=nli_result.support_ratio,
            poison_score=poison_result.overall_probability
        )
        
        # Return complete evaluation
        return EvaluationResult(
            trust_score=trust_result.trust_score,
            trust_level=trust_result.trust_level,
            is_trustworthy=trust_result.is_trustworthy,
            nli_verification=nli_result,
            poison_detection=poison_result,
            detailed_report=self._generate_report(...)
        )
```

**What you get back:**

```python
result = agent.evaluate(query, response, documents)

print(result.trust_score)      # 0.89
print(result.trust_level)      # TrustLevel.HIGH
print(result.is_trustworthy)   # True
print(result.summary)          # "Response is TRUSTWORTHY..."
print(result.detailed_report)  # Full formatted report
```

---

### How the Evaluation Agent Works (Visual Flow)

```
                    QUERY + RESPONSE + DOCUMENTS
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     EVALUATION AGENT                         │
│                                                              │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐    │
│  │ NLI VERIFIER │   │   POISON     │   │  TRUST INDEX │    │
│  │              │   │  DETECTOR    │   │  CALCULATOR  │    │
│  │ Factuality:  │   │              │   │              │    │
│  │    0.90      │   │ Poison: 0.10 │   │ Combines all │    │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘    │
│         │                  │                   │            │
│         └────────────┬─────┴───────────────────┘            │
│                      ▼                                       │
│              ┌───────────────┐                              │
│              │ TRUST SCORE:  │                              │
│              │    0.89       │                              │
│              │ Level: HIGH   │                              │
│              └───────────────┘                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    EVALUATION RESULT
                 (trust score + report)
```

---

## 8. Configuration

### What is `config.yaml`?

A file that stores all settings in one place. Instead of changing code, you change the config file.

**Current settings:**

```yaml
# Which embedding model to use for converting text to numbers
EMBEDDING_MODEL: "sentence-transformers/all-MiniLM-L6-v2"

# Which LLM provider and model
LLM_PROVIDER: "openai_compatible"
LLM_MODEL: "llama3.3:70b"

# FARMI connection
FARMI_API_URL: "https://gptlab.rd.tuni.fi/students/ollama/v1"
FARMI_API_KEY: "sk-ollama-..."

# NLI model for fact checking
NLI_MODEL: "facebook/bart-large-mnli"

# Document chunking settings
CHUNK_SIZE: 512        # Max characters per chunk
CHUNK_OVERLAP: 50      # Characters to overlap between chunks
TOP_K_RETRIEVAL: 5     # How many documents to retrieve

# Trust Index weights
TRUST_ALPHA: 0.4       # Factuality weight
TRUST_BETA: 0.35       # Consistency weight
TRUST_GAMMA: 0.25      # Poisoning weight
```

---

## 9. How to Use

### Basic Usage (Without Evaluation)

```python
from src.pipeline import RAGPipeline

# 1. Create the pipeline with config
config = {
    'FARMI_API_URL': 'https://gptlab.rd.tuni.fi/students/ollama/v1',
    'FARMI_API_KEY': 'your-api-key',
    'LLM_MODEL': 'llama3.3:70b',
    'NLI_MODEL': 'facebook/bart-large-mnli',
    'TRUST_THRESHOLD': 0.5,
    # ... other settings
}

# Disable evaluation for faster queries (no trust check)
pipeline = RAGPipeline(config=config, enable_evaluation=False)

# 2. Add your documents (knowledge base)
pipeline.add_documents([
    "Paris is the capital of France.",
    "London is the capital of the UK.",
    "Tokyo is the capital of Japan."
])

# 3. Ask questions!
result = pipeline.query("What is the capital of France?")

print(f"Question: {result.query}")
print(f"Answer: {result.response}")
print(f"Sources: {result.retrieved_docs}")
```

### Usage with Evaluation (Recommended)

```python
from src.pipeline import RAGPipeline

# Create pipeline with evaluation enabled
pipeline = RAGPipeline(config=config, enable_evaluation=True)

# Add documents
pipeline.add_documents([
    "Paris is the capital of France.",
    "The Eiffel Tower is located in Paris."
])

# Query with evaluation - gets trust score!
result = pipeline.query_with_evaluation("What is the capital of France?")

# Access the answer
print(f"Answer: {result.response}")

# Access trust information
print(f"Trust Score: {result.trust_score:.2f}")  # e.g., 0.85
print(f"Trust Level: {result.trust_level.value}")  # e.g., "high"
print(f"Is Trustworthy: {result.is_trustworthy}")  # True or False

# Print detailed evaluation report
pipeline.print_evaluation_report(result)
```

**Sample Output:**
```
Answer: The capital of France is Paris.

Trust Score: 0.85
Trust Level: high
Is Trustworthy: True

============================================================
EVALUATION AGENT REPORT
============================================================

QUERY:
  What is the capital of France?

RESPONSE:
  The capital of France is Paris.

------------------------------------------------------------
TRUST INDEX
------------------------------------------------------------
  Overall Score: 0.85 / 1.00
  Trust Level: HIGH
  Is Trustworthy: YES

  Component Scores:
    - Factuality:    0.92 (contributes 0.368)
    - Consistency:   0.88 (contributes 0.308)
    - Poison Safety: 0.95 (contributes 0.238)

------------------------------------------------------------
RECOMMENDATION
------------------------------------------------------------
  This response appears trustworthy. The answer is well-supported
  by consistent, clean documents.
============================================================
```

### Loading Datasets

```python
# Load FEVER dataset
pipeline.load_fever_data("data/raw/fever/fever_train.jsonl")

# Load TruthfulQA dataset
pipeline.load_truthfulqa_data("data/raw/truthfulqa/truthfulqa.jsonl")
```

---

## 10. Glossary

| Term | Simple Definition |
|------|-------------------|
| **RAG** | Retrieval-Augmented Generation - AI that searches documents before answering |
| **Embedding** | Text converted to numbers so computers can compare them |
| **Vector** | A list of numbers (like [0.23, -0.45, 0.67]) |
| **FAISS** | A fast library for searching similar vectors |
| **LLM** | Large Language Model - AI that understands and generates text |
| **Prompt** | Instructions and context given to an LLM |
| **NLI** | Natural Language Inference - checks if text A supports/refutes text B |
| **Trust Index** | A score (0-1) indicating how trustworthy an answer is |
| **Poisoning** | Injecting false information to trick the AI |
| **FARMI** | Tampere University's AI computing cluster |
| **API** | Application Programming Interface - a way for programs to talk to each other |
| **Token** | A piece of text (roughly a word or part of a word) |
| **Chunk** | A small piece of a larger document |

---

## Changelog

### Version 0.2.0 (February 2026)

**Completed:**
- ✅ Retriever module (embeddings, document processing, vector store)
- ✅ Generator module (FARMI client, prompts, generator)
- ✅ Pipeline module (basic RAG pipeline)
- ✅ Configuration system
- ✅ **Evaluation Agent (NEW!)**
  - NLI Verifier for factual consistency checking
  - Poison Detector for adversarial content detection
  - Trust Index Calculator for composite trust scores
  - Full integration with RAG pipeline

**Next:**
- ⏳ Testing with FEVER and TruthfulQA datasets
- ⏳ Experiment framework for evaluation
- ⏳ Performance optimization

### Version 0.1.0 (February 2026)

**Initial Release:**
- Retriever module
- Generator module
- Basic pipeline
- Configuration system

---

*This documentation will be updated as we continue building the project.*
