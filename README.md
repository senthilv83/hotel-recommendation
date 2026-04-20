# 🏨 Hotel Operations GenAI Assistant

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![Agentic AI](https://img.shields.io/badge/AI-Agentic_Framework-orange.svg)
![Graph RAG](https://img.shields.io/badge/RAG-Graph_RAG-green.svg)
![Semantic Search](https://img.shields.io/badge/NLP-Semantic_Search-purple.svg)
![Knowledge Graph](https://img.shields.io/badge/Data-Knowledge_Graph-red.svg)
![Hugging Face](https://img.shields.io/badge/LLM-Hugging_Face-yellow.svg)

**Transforming Fragmented Maintenance Logs into Policy-Driven Intelligence using Semantic RAG, Graph RAG, and Agentic AI**

---

## ⚠️ The Problem Statement: The "Contextual Blindness" of Tabular Systems
In the high-stakes hospitality industry (e.g., Marriott, Hilton), operational intelligence is often buried in thousands of isolated, tabular maintenance logs. 

**The Challenge:**
When a guest in **Room 402** reports an **AC failure**, traditional Property Management Systems (PMS) treat it as a discrete, one-off event. This creates a state of **"Contextual Blindness"**:
*   **Semantic Fragmentation:** A "warm mini-fridge" and an "AC cooling failure" are logged as unrelated hardware tickets, even though they represent the same systemic failure: **Thermal Control Regulation.**
*   **Operational Memory Loss:** Systems lack the topological awareness to link today's AC issue to last month's fridge failure, resulting in "band-aid" repairs rather than identifying deep-seated infrastructure hotspots.
*   **Policy Enforcement Gap:** Front-desk staff often fail to trigger complex corporate guest-satisfaction policies (e.g., Policy #7: Deep technical audits for repeat failures) because they cannot "connect the dots" across siloed data points.

**The Solution:**
This project introduces a **Tri-Layered GenAI Framework** (Semantic RAG + Graph RAG + Agentic AI) that converts fragmented logs into a **Knowledge Graph**, revealing systemic hotspots and automating corporate policy execution through autonomous reasoning.

---

## 🔬 1. Why this Research Approach is Promising for the Enterprise
We selected a hybrid methodology because it uniquely solves the scaling and reasoning limitations of standard AI implementations:

1. **Semantic RAG (Vector Embeddings):** Overcomes human vocabulary variation. It captures the *intent* of a ticket, eliminating the need for strict, drop-down taxonomy systems that staff rarely use correctly.
2. **Graph RAG (Topological Relationships):** Traditional RAG (Vector Search) is terrible at connecting multi-hop relationships over time. By mapping operational data into a **Knowledge Graph**, the system inherently gains *memory*. It maps seemingly unrelated issues (AC vs. Fridge) to a shared semantic node ("Thermal Control Regulation"). This shifts maintenance from **reactive** (fixing one AC) to **predictive** (auditing a room's thermal grid).
3. **Agentic AI (Autonomous Reasoning):** Rather than just presenting a dashboard to a human, we use an open-source Small Language Model (SLM) to reason over the Graph RAG context and apply corporate policies automatically.

**Enterprise Scalability:** This approach minimizes AI hallucinations by grounding the LLM entirely in deterministic Knowledge Graphs and hardcoded corporate manuals, making it safe for production-scale operational deployment.

---

## 📓 2. Step-by-Step Technical Explanation of the Notebook
The `Hotel_Ops_GenAI.ipynb` notebook is designed for storytelling. It walks stakeholders from the problem (raw data) to the solution (Autonomous Action).

### 🛠️ Cell 1: Data Ingestion & Operational Baseline
* **Why this step:** To establish the baseline problem. It proves the difficulty of spotting patterns in standard CSV logs.
* **Input:** `data/maintenance_logs.csv` (100 rows of synthetic logs).
* **Output:** A Pandas DataFrame view and a custom Matplotlib bar chart visually highlighting "AC failure" and "Mini-fridge warm" inside a red "Thermal Control Regulation Bucket."

### 🧠 Cell 2: Semantic Retrieval (Beyond Keywords)
* **Why this step:** To demonstrate that keyword matching is fundamentally flawed for human-generated text. 
* **Input:** A natural language query (`"Temperature Control thermal issues"`) and ChromaDB vector store.
* **Output:** The Hugging Face embedding model (`all-MiniLM-L6-v2`) successfully groups BOTH the AC failure and the Mini-fridge warm issues across multiple rooms into a single semantic result set.

### 🕸️ Cell 3: Deep Dive - Keyword Matching vs. Graph RAG (Single Entry)
* **Why this step:** To visually prove why topological context (Graph RAG) is superior to isolated database lookups.
* **Input:** A single new ticket: *"AC failure in Room 402"*.
* **Output:** A NetworkX graph connecting the Room (Blue), the Current Issue (Red), Past Issues (Gray), and the overarching **Semantic Node: Temperature Control** (Orange). This proves the system "remembers" context.

### 🌐 Cell 4: Macro Graph Relationships (Spotting Hotspots)
* **Why this step:** To prove enterprise scalability. It maps the entire property's issues to identify systemic "hotspots."
* **Input:** The complete maintenance log dataset.
* **Output:** A high-density visual map. Rooms 402 and 504 are colored Hot Pink and linked directly to a massive Orange Semantic Node representing "Thermal Regulation," immediately drawing executive attention to the systemic failure.

### 🤖 Cell 5: Agentic AI Recommendation Engine
* **Why this step:** Enterprise value is created through action. This step uses an Agentic SLM to process the Graph RAG data and execute policy.
* **Input:** 
  1. The specific Graph RAG context for Room 402 (Current AC failure + Past Fridge failure).
  2. The raw text of Marriott Corporate Policies.
* **Output:** A strict, deterministic, 2-bullet point operational directive generated by `Qwen1.5-0.5B-Chat`:
  * **Recommendation:** Take the room out of service and offer the guest a 20% discount.
  * **Hotel Policy:** Explicit citation of Policies #1 and #7.

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
