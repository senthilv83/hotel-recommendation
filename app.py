import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
import os

st.set_page_config(page_title="Hotel Ops GenAI Assistant", layout="wide")

st.title("🏨 Hotel Operations GenAI Assistant")
st.markdown("### Leveraging GenAI for Operational Intelligence")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("data/maintenance_logs.csv")
    with open("data/hotel_policies.txt", "r") as f:
        policies = f.read()
    return df, policies

df, policies = load_data()

# Sidebar
st.sidebar.header("System Configuration")
retrieval_mode = st.sidebar.radio("Retrieval Mode", ["Key-based matching", "Semantic Retrieval (RAG)", "Graph RAG"])

# Semantic Engine Setup
@st.cache_resource
def setup_vector_store(_df):
    documents = []
    for _, row in _df.iterrows():
        content = f"Room {row['Room']} had {row['Issue']} on {row['Date']}. Status: {row['Status']}"
        documents.append(Document(page_content=content, metadata={"room": str(row['Room']), "date": row['Date']}))
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents, embeddings, persist_directory="./chroma_db")
    return vectorstore

vectorstore = setup_vector_store(df)

# User Input
query_room = st.text_input("Enter Room Number (e.g., 402):", "402")
user_query = st.text_input("Ask a question about this room:", f"What maintenance issues happened in Room {query_room}?")

if st.button("Run Intelligence Analysis"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Retrieval Results")
        if retrieval_mode == "Key-based matching":
            results = df[df['Room'] == int(query_room)]
            st.dataframe(results)
            retrieved_text = results.to_string()
        
        elif retrieval_mode == "Semantic Retrieval (RAG)":
            search_query = f"Issues in room {query_room}"
            docs = vectorstore.similarity_search(search_query, k=5)
            retrieved_text = "\n".join([doc.page_content for doc in docs])
            for doc in docs:
                st.info(doc.page_content)
        
        elif retrieval_mode == "Graph RAG":
            st.warning("Graph RAG: Showing relationships between Room, Issue, and Status")
            G = nx.Graph()
            room_node = f"Room {query_room}"
            G.add_node(room_node, color='red')
            
            room_issues = df[df['Room'] == int(query_room)]
            for _, row in room_issues.iterrows():
                issue_node = row['Issue']
                G.add_edge(room_node, issue_node)
            
            fig, ax = plt.subplots()
            nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', ax=ax)
            st.pyplot(fig)
            retrieved_text = room_issues.to_string()

    with col2:
        st.subheader("🤖 GenAI Recommendation")
        # Simulating LLM response based on Mistral logic
        # In a real app, we'd call the Mistral API or local model
        st.write("**Context retrieved from Hotel Policies:**")
        st.caption(policies[:200] + "...")
        
        prompt = f"""
        Context: {retrieved_text}
        Policies: {policies}
        User Query: {user_query}
        """
        
        # Mocking the LLM Response for demo purposes
        st.success("Analysis Complete!")
        if "402" in query_room:
            st.markdown("""
            **Recommendation for Room 402:**
            - **History:** Multiple AC failures and plumbing issues logged in early 2026.
            - **Policy Alignment:** According to Marriott Policy #7 (Repeat Issues), Room 402 has exceeded 3 issues in a month.
            - **Action:** **Take Room 402 out of service immediately** for a deep technical audit.
            - **Guest Recovery:** Offer the next guest in 402 a complimentary breakfast voucher as per Policy #6.
            """)
        else:
            st.markdown(f"Based on the logs for Room {query_room}, the issues are within normal parameters. Ensure 'Pending' tasks are assigned to the morning shift.")

# Data Visuals
st.divider()
st.subheader("📊 Operational Trends (2025-2026)")
fig2, ax2 = plt.subplots(figsize=(10, 4))
df['Issue'].value_counts().plot(kind='bar', ax=ax2, color='skyblue')
plt.title("Frequency of Maintenance Issues")
st.pyplot(fig2)
