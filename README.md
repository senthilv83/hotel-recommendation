# 🏨 Hotel Operations GenAI Assistant
**Transforming Maintenance Logs into Policy-Driven Intelligence using Agentic AI, Semantic RAG, and Graph RAG**

## 📖 The Vision
Hotels generate thousands of data points daily—maintenance logs, guest complaints, and work orders. Traditionally, this data sits in tabular databases requiring manual, keyword-based searches. If a guest complains about a "warm room," a standard system searching for "AC failure" will miss it. Furthermore, traditional systems treat every ticket as an isolated event, missing critical systemic failures.

This project introduces an **Agentic AI Framework** that bridges the gap between raw operational data and corporate policy execution.

## 🔬 Research Approach & Enterprise Advantages
Why combine **Semantic RAG**, **Graph RAG**, and **Agentic AI**?

1. **Semantic Understanding (Beyond Keywords):** Using vector embeddings, the system understands the *intent* behind a ticket. It maps a "Mini-fridge warm" issue and an "AC failure" issue to the same semantic concept: *Thermal Regulation*.
2. **Topological Context (Graph RAG):** By converting tabular logs into a Knowledge Graph (Nodes = Rooms/Issues, Edges = Occurrences), the AI instantly visualizes "hotspots." It gives the AI *memory* of a room's history, rather than treating a new ticket in a vacuum.
3. **Automated Policy Execution (Agentic AI):** Instead of just providing a dashboard, an open-source Hugging Face Small Language Model (SLM) is used as an autonomous reasoning engine. It reads the Graph RAG context, cross-references it with corporate policies (e.g., Marriott Policy #1 & #7), and issues a strict operational directive.

### 🚀 Advantages for Enterprise Scale
* **Proactive Maintenance:** Identify failing infrastructure (e.g., room electrical grids) before multiple guests are impacted.
* **Automated Compliance:** Ensure front-desk and maintenance staff adhere to complex corporate compensation policies without having to memorize them.
* **Data-Agnostic Scaling:** The framework can scale from 100 logs to millions of cross-property logs without altering the core reasoning engine.

---

## 📓 Storytelling: Jupyter Notebook Breakdown
The `Hotel_Ops_GenAI.ipynb` notebook is structured as an executive storytelling demo.

### Intro: The Agentic AI Promise
Sets the stage: Moving from isolated tickets to connected, automated actions.

### Step 1: Data Exploration (The Problem)
Loads raw, synthetic tabular data (2025-2026). Visualizes the frequency of issues.
* **The "Why":** Demonstrates how hard it is for humans to spot long-term patterns in raw `.csv` files.

### Step 2: Semantic Retrieval
Converts logs into vector embeddings using ChromaDB and Hugging Face (`all-MiniLM-L6-v2`). 
* **The "Why":** Proves the system can find "AC Failures" even when the user searches for "Temperature problems."

### Deep Dive: Keyword Matching vs. Graph RAG
Simulates a single new ticket arriving for Room 402.
* **Keyword View:** Shows 1 isolated row. No context.
* **Graph RAG View:** Renders a visual NetworkX graph mapping the current AC issue AND past mini-fridge issues to a shared **Semantic Node: Temperature Control**.
* **The "Why":** Visually proves to stakeholders why context matters.

### Step 3: Graph Relationships (Hotspots)
Generates a full Knowledge Graph of the entire hotel.
* **The "Why":** Shows structural hotspots where preventative maintenance should be prioritized.

### Step 4: Agentic AI Recommendation Framework
Injects the Graph RAG Context and raw Hotel Policies into an open-source Instruction-Tuned LLM (`Qwen1.5-0.5B-Chat` acting as a fast local stand-in for Mistral). The Agent is strictly prompted to avoid conversational fluff.
* **The "Why":** Demonstrates autonomous reasoning. The Agent outputs a crisp, 2-bullet point directive (Recommendation & Hotel Policy) dictating room audits and guest compensation (20% discount).

---

## 🛠️ Tech Stack
* **Orchestration:** LangChain
* **Vector Database:** ChromaDB
* **Embeddings & LLM:** Hugging Face Transformers (`Qwen/Qwen1.5-0.5B-Chat`)
* **Knowledge Graph:** NetworkX, Matplotlib
* **Data Manipulation:** Pandas

## 💻 How to Run Locally
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the interactive Streamlit UI:
   ```bash
   streamlit run app.py
   ```
4. Explore the Storytelling Notebook:
   ```bash
   jupyter notebook Hotel_Ops_GenAI.ipynb
   ```
