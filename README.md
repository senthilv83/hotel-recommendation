# 🏨 Hotel Operations GenAI Assistant

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![Agentic AI](https://img.shields.io/badge/AI-Agentic_Framework-orange.svg)
![Graph RAG](https://img.shields.io/badge/RAG-Graph_RAG-green.svg)
![Semantic Search](https://img.shields.io/badge/NLP-Semantic_Search-purple.svg)
![Knowledge Graph](https://img.shields.io/badge/Data-Knowledge_Graph-red.svg)
![Hugging Face](https://img.shields.io/badge/LLM-Hugging_Face-yellow.svg)

**Transforming Maintenance Logs into Policy-Driven Intelligence using Agentic AI, Semantic RAG, and Graph RAG**

---

## 🔬 1. Why this Research Approach is Promising for the Enterprise
In large-scale hospitality (e.g., Marriott, Hilton), raw operational data is siloed in tabular databases. Traditional systems rely on exact keyword matches, meaning a complaint about a "warm fridge" and an "AC failure" are treated as completely distinct, isolated incidents.

### The Methodology: A Tri-layered Approach
We selected a hybrid methodology—**Semantic RAG + Graph RAG + Agentic AI**—because it uniquely solves the scaling and reasoning limitations of standard AI implementations:

1. **Semantic RAG (Vector Embeddings):** Overcomes human vocabulary variation. It captures the *intent* of a ticket. This is promising because it eliminates the need for strict, drop-down taxonomy systems that staff rarely use correctly.
2. **Graph RAG (Topological Relationships):** Traditional RAG (Vector Search) is terrible at connecting multi-hop relationships over time. By mapping operational data into a **Knowledge Graph**, the system inherently gains *memory*. It maps seemingly unrelated issues (AC vs. Fridge) to a shared semantic node ("Temperature Control"). This is highly promising for the enterprise because it shifts maintenance from **reactive** (fixing one AC) to **proactive/predictive** (auditing a room's thermal/electrical grid).
3. **Agentic AI (Autonomous Reasoning):** Rather than just presenting a dashboard to a human, we use an open-source Small Language Model (SLM) to reason over the Graph RAG context and apply corporate policies. This guarantees that complex guest-compensation policies (e.g., "20% discount if unresolved in 2 hours") are uniformly enforced without relying on front-desk staff memory.

**Enterprise Scalability:** This approach minimizes AI hallucinations by grounding the LLM entirely in deterministic Knowledge Graphs and hardcoded corporate manuals, making it safe for production-scale operational deployment.

---

## 📓 2. Step-by-Step Technical Explanation of the Notebook

The `Hotel_Ops_GenAI.ipynb` notebook is designed for storytelling. It walks stakeholders from the problem (raw data) to the solution (Autonomous Action).

### 🛠️ Cell 1: Data Ingestion & Exploration
* **Why this step:** To establish the baseline problem. Humans and traditional systems struggle to identify recurring patterns inside raw, tabular CSV files.
* **Input:** `data/maintenance_logs.csv` (100 rows of synthetic room numbers, issues, dates, and statuses).
* **Output:** A Pandas DataFrame view and a Matplotlib bar chart showing the frequency of total issues across the hotel.

### 🧠 Cell 2: Semantic Retrieval (Beyond Keywords)
* **Why this step:** To demonstrate that keyword matching is fundamentally flawed for human-generated text. 
* **Input:** A natural language query (`"Temperature problems in room 402"`) and ChromaDB vector store.
* **Output:** The system retrieves logs like "Mini-fridge warm," proving the Hugging Face embedding model (`all-MiniLM-L6-v2`) understands semantic *meaning* rather than just string matching.

### 🕸️ Cell 3: Deep Dive - Keyword Matching vs. Graph RAG (Single Entry)
* **Why this step:** To visually prove why topological context (Graph RAG) is superior to isolated database lookups.
* **Input:** A single new ticket: *"AC failure in Room 402"*.
* **Output:** Two distinct views. First, the isolated tabular row. Second, a NetworkX graph connecting the Room (Blue), the Current Issue (Red), Past Issues (Gray), and the overarching **Semantic Node: Temperature Control** (Orange). This proves the system "remembers" context.

### 🌐 Cell 4: Macro Graph Relationships (Spotting Hotspots)
* **Why this step:** To show how this scales to the entire property. By graphing all rooms and all issues, structural hotspots become glaringly obvious.
* **Input:** The entire maintenance log dataset.
* **Output:** A macro-level Knowledge Graph (NetworkX) where heavily connected nodes (problematic rooms) draw immediate attention for preventative maintenance.

### 🤖 Cell 5: Agentic AI Recommendation Engine
* **Why this step:** Data visualization is not enough; enterprise value is created through action. This step uses an Agentic SLM to process the Graph RAG data and execute policy.
* **Input:** 
  1. The specific Graph RAG context for Room 402 (Current AC failure + Past Fridge failure = Temperature Control issue).
  2. The raw text of Marriott Corporate Policies (#1 and #7).
* **Output:** A strict, deterministic, 2-bullet point operational directive:
  * **Recommendation:** Action for the room and guest compensation directives.
  * **Hotel Policy:** The explicit citation of the rules justifying the action.

---

## 💻 3. Tech Stack & Execution
* **Language:** Python 3.10+
* **AI Orchestration:** Hugging Face `transformers`, `torch`
* **Local SLM:** `Qwen/Qwen1.5-0.5B-Chat` (Stand-in for larger Mistral/Llama models)
* **Vector DB:** ChromaDB + `sentence-transformers`
* **Knowledge Graph:** `networkx`, `matplotlib`
* **Data Handling:** `pandas`

### How to Run Locally
```bash
git clone https://github.com/senthilv83/hotel-recommendation.git
cd hotel-recommendation
pip install -r requirements.txt
jupyter notebook Hotel_Ops_GenAI.ipynb
```
