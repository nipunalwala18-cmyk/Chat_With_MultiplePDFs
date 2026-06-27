import os
import re
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_community.vectorstores import FAISS

from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from langchain.tools import tool


# ---------------- PDF Processing ---------------- #

def get_pdf_text(pdf_docs):
    text = ""

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)

        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text


def get_chunks_text(raw_text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    return splitter.split_text(raw_text)


# ---------------- Vector Store ---------------- #

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
        # Replace with the model returned by models.list() if needed.
    )

    vector_store = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings,
    )

    return vector_store


# ---------------- Agent ---------------- #

def get_agent(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    @tool
    def search_pdf(query: str) -> str:
        """Search the uploaded PDF documents."""
        docs = retriever.invoke(query)

        return "\n\n".join(doc.page_content for doc in docs)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
    )

    memory = InMemorySaver()

    agent = create_agent(
        model=llm,
        tools=[search_pdf],
        checkpointer=memory,
        system_prompt=(
            "You are a helpful assistant. "
            "Always search the uploaded PDFs before answering. "
            "If the answer is not present in the PDFs, say you couldn't find it."
        ),
    )

    return agent


# ---------------- Streamlit ---------------- #

def main():
    load_dotenv()

    st.set_page_config(page_title="Chat with Multiple PDFs")

    if "agent" not in st.session_state:
        st.session_state.agent = None

    st.header("Chat with Multiple PDFs")

    user_question = st.text_input(
        "Ask a question about your documents"
    )

    with st.sidebar:
        st.subheader("Your Documents")

        pdf_docs = st.file_uploader(
            "Upload your PDFs",
            accept_multiple_files=True,
        )

        if st.button("Process"):

            if not pdf_docs:
                st.warning("Please upload at least one PDF.")
                st.stop()

            with st.spinner("Processing..."):

                raw_text = get_pdf_text(pdf_docs)

                chunks = get_chunks_text(raw_text)

                vector_store = get_vector_store(chunks)

                st.session_state.agent = get_agent(vector_store)

                st.success("Documents processed successfully!")

    if user_question:

        if st.session_state.agent is None:
            st.warning("Please upload and process PDFs first.")
            return

        response = st.session_state.agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": user_question,
                    }
                ]
            },
            config={
                "configurable": {
                    "thread_id": "streamlit-user"
                }
            },
        )

        assistant_message = response["messages"][-1]

        text = ""

        for block in assistant_message.content:
            if block["type"] == "text":
                text += block["text"]

        st.write(text)
        
if __name__ == "__main__":
    main()
