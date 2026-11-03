# IT Helpdesk & Incident Resolution Assistant using Hybrid RAG

An AI-powered IT Helpdesk Assistant that helps users solve common IT problems by retrieving information from trusted technical PDF documents and generating clear, step-by-step troubleshooting instructions.

##  Project Overview

IT support teams frequently handle repeated technical issues such as:

- DNS resolution errors
- Printer connectivity problems
- Remote Desktop connection issues
- Windows Update errors
- Network troubleshooting
- Security and incident response questions

Searching through lengthy technical documents to find solutions can be time-consuming.

To address this problem, I developed an **IT Helpdesk Assistant using Retrieval-Augmented Generation (RAG)**. It allows users to ask IT-related questions and receive answers based on a collection of trusted technical documents.

The application uses Hybrid Retrieval, combining keyword-based search and meaning-based search, followed by reranking to select the most relevant information.

## Project Objectives

- Provide quick answers to common IT troubleshooting questions.
- Retrieve information from trusted technical PDF documents.
- Combine keyword search and vector search for better retrieval.
- Generate clear, numbered troubleshooting steps.
- Display source documents and page references.
- Avoid unsupported answers using a safe fallback response.

##  System Architecture

The project consists of two main stages:

### 1. Indexing – One-Time Process

1. Collect technical PDF documents.
2. Load documents using PyPDFLoader.
3. Split documents into smaller chunks.
4. Generate embeddings using OpenAI.
5. Store embeddings in ChromaDB.
6. Prepare the BM25 keyword search index.

### 2. Question Answering – Every User Query

1. User enters an IT-related question.
2. BM25 retrieves keyword-matching chunks.
3. ChromaDB retrieves meaning-related chunks.
4. EnsembleRetriever combines both retrieval methods.
5. Cross-Encoder reranks the retrieved chunks.
6. The top 4 relevant chunks are selected.
7. GPT-4.1-mini generates an answer using the retrieved context.
8. The application displays the answer and source references.

## Why Hybrid RAG?

Hybrid RAG combines two retrieval approaches:

| Retrieval Method | Purpose |
|---|---|
| BM25 | Finds exact keywords, commands and error names |
| Vector Search | Finds information based on meaning |
| EnsembleRetriever | Combines both retrieval methods |
| Cross-Encoder Reranker | Ranks retrieved information by relevance |

**Retrieval configuration:**
- BM25 retriever: Top 6 results
- Chroma vector retriever: Top 6 results
- Ensemble weights: 0.5 / 0.5
- Reranking: Cross-Encoder
- Final context: Top 4 chunks

## 📚 Data Collection

A total of **22 PDF documents containing 119 pages** were collected from official technical documentation sources.

| Source | Number of PDFs |
|---|---:|
| Microsoft Learn | 15 |
| Microsoft Support | 1 |
| NIST / CSRC | 2 |
| OWASP | 1 |
| Ubuntu Documentation | 2 |
| Cisco Documentation | 1 |
| **Total** | **22** |

These documents cover DNS, DHCP, Remote Desktop, printing, Windows troubleshooting, networking, API security, Linux and incident response.

## ⚙️ Knowledge Base Preparation

| Component | Configuration |
|---|---|
| PDF Loader | PyPDFLoader |
| Text Splitter | RecursiveCharacterTextSplitter |
| Chunk Size | 1000 |
| Chunk Overlap | 200 |
| Total Chunks | 240 |
| Embedding Model | OpenAI text-embedding-3-large |
| Vector Database | ChromaDB |

The chunk overlap helps preserve related information when documents are divided into smaller pieces.

The Chroma vector database is persisted and reused when the application runs again.

## Answer Generation & Prompt Design

**Language Model:** GPT-4.1-mini

The prompt includes the following instructions:

- Answer only using the retrieved context.
- Provide clear numbered troubleshooting steps.
- Limit the answer to a maximum of 8 steps.
- Copy technical commands exactly as written in the documents.
- Avoid unsupported information.
- If relevant information is unavailable, return:

`I could not find this in the knowledge base.`

Source documents and page numbers are displayed separately using document metadata.

##  Model Evaluation

The system was evaluated using **Ragas** with 15 test questions covering DNS, printers, Remote Desktop, stop errors and incident response.

| Evaluation Metric | Score | Purpose |
|---|---:|---|
| Faithfulness | 0.82 | Checks whether answers are supported by retrieved context |
| Answer Relevancy | 0.93 | Checks whether answers address the questions |
| Context Precision | 0.90 | Checks the relevance of retrieved chunks |
| Context Recall | 0.90 | Checks whether the required information was retrieved |

These evaluation results help assess the quality of retrieval and answer generation.

##  Streamlit Web Application

The application provides:

-  Interactive question-and-answer interface.
-  PDF document upload.
-  Hybrid retrieval and reranking.
-  AI-generated troubleshooting instructions.
-  Source documents and page references.
-  Safe fallback for questions outside the knowledge base.
-  Persistent ChromaDB index.

### Application Workflow

User Question → Hybrid Retrieval → Reranking → GPT-4.1-mini → Answer with Sources

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| LLM | GPT-4.1-mini |
| Embeddings | OpenAI text-embedding-3-large |
| Retrieval | BM25, Chroma Vector Search |
| Retriever Combination | LangChain EnsembleRetriever |
| Reranking | Cross-Encoder |
| Document Processing | PyPDFLoader |
| Text Splitting | RecursiveCharacterTextSplitter |
| Vector Database | ChromaDB |
| Evaluation | Ragas |

## 🚀 Installation & Setup

### Prerequisites

- Python 3.11
- OpenAI API key
- Git
- Internet connection for API access

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

```bash
cd YOUR_REPOSITORY
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

Create a `.env` file in the project directory.

```env
OPENAI_API_KEY=your_openai_api_key
```

Replace the placeholder with your own API key.

**Important:** Never upload your `.env` file or expose your API key on GitHub.

### Step 5: Add Knowledge Base Documents

Place the required technical PDF documents in the configured knowledge base folder.

Update the document folder path in the application if your local folder location is different.

### Step 6: Run the Application

```bash
streamlit run app.py
```

The Streamlit application will open in your browser.

## Example Questions

Try asking the assistant:

1. How do I troubleshoot DNS resolution problems on a Windows computer?
2. What should I check when a printer is not responding?
3. How can I troubleshoot Remote Desktop connection problems?
4. What should I do when Windows Update fails?
5. How do I troubleshoot network connectivity issues?

## 🔮 Future Scope

- Improve retrieval quality for incident response questions.
- Add more technical sources, including internal runbooks and security playbooks.
- Deploy the application to the cloud with user authentication.
- Integrate the assistant with IT ticketing systems.

## Key Highlights

- 22 technical PDF documents.
- 119 pages of technical documentation.
- 240 text chunks.
- Hybrid retrieval using BM25 and ChromaDB.
- Cross-Encoder-based reranking.
- GPT-4.1-mini for answer generation.
- Ragas evaluation using 15 test questions.
- Streamlit-based interactive application.

## Author

**Tejaswi Siddagoni**

B.Tech – Artificial Intelligence and Machine Learning

Innomatics Research Labs – Data Science Program

## Acknowledgement

Developed as part of the Data Science learning program at Innomatics Research Labs.

---

⭐ If you find this project useful, feel free to explore the repository!
