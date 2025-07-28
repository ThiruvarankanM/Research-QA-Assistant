import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_community.retrievers import ArxivRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from PyPDF2 import PdfReader
import tempfile

load_dotenv(override=True)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

st.set_page_config(page_title="Academic Research Q&A Assistant", layout="wide")

st.title("Academic Research Q&A Assistant")
st.markdown(
    "Ask questions about multiple arXiv research papers or upload your own PDF paper."
)

# Sidebar settings
with st.sidebar:
    st.header("Load Papers")

    arxiv_codes_input = st.text_area(
        "Enter arXiv IDs (comma separated)", help="Example: 2102.00001, 2303.01234"
    )
    load_arxiv_button = st.button("Load arXiv Papers")

    st.markdown("---")

    st.header("Upload Your Paper")
    uploaded_pdf = st.file_uploader(
        "Upload PDF file (optional)", type=["pdf"], help="Upload your own research paper"
    )
    load_pdf_button = st.button("Load Uploaded PDF")

# Session state to store loaded documents and metadata
if "papers" not in st.session_state:
    st.session_state.papers = []
    st.session_state.paper_summaries = []
    st.session_state.current_context = ""
    st.session_state.chat_history = []

# Show guidance if no input is given
if not arxiv_codes_input.strip() and uploaded_pdf is None:
    st.info("👋 To begin, please enter arXiv IDs on the left or upload a PDF paper.")


def summarize_text(text):
    prompt = ChatPromptTemplate.from_template(
        "Summarize the following research paper abstract or content briefly:\n\n{text}"
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke(text)

def load_arxiv_papers(arxiv_codes):
    docs = []
    for code in arxiv_codes:
        retriever = ArxivRetriever(
            load_max_docs=1,
            doc_ids=[code],
            get_full_documents=True,
        )
        paper_docs = retriever.invoke(code)
        docs.extend(paper_docs if isinstance(paper_docs, list) else [paper_docs])
    return docs


def load_pdf_content(file) -> str:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n\n"
    return text

# Load arXiv papers
if load_arxiv_button and arxiv_codes_input.strip():
    codes = [code.strip() for code in arxiv_codes_input.split(",") if code.strip()]
    with st.spinner(f"Loading {len(codes)} paper(s) from arXiv..."):
        docs = load_arxiv_papers(codes)
    st.session_state.papers = docs
    # Summarize all loaded papers
    st.session_state.paper_summaries = [
        summarize_text(doc.page_content[:2000]) for doc in docs
    ]
    st.session_state.current_context = "\n\n".join(doc.page_content for doc in docs)
    st.session_state.chat_history = []
    st.success(f"Loaded and summarized {len(codes)} paper(s).")

# Load uploaded PDF
if load_pdf_button and uploaded_pdf is not None:
    with st.spinner("Extracting text from PDF..."):
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_pdf.read())
            tmp_file_path = tmp_file.name

        pdf_text = load_pdf_content(tmp_file_path)
        st.session_state.papers = [pdf_text]
        st.session_state.paper_summaries = [summarize_text(pdf_text[:2000])]
        st.session_state.current_context = pdf_text
        st.session_state.chat_history = []
    st.success("Uploaded PDF loaded and summarized.")

# Show loaded papers info
if st.session_state.papers:
    st.markdown("---")
    st.header("Loaded Papers Summary")
    for i, summary in enumerate(st.session_state.paper_summaries, 1):
        st.markdown(f"**Paper {i} Summary:** {summary}")

    st.markdown("---")
    st.header("Ask a Question")
    question = st.text_input("Type your question related to the loaded papers:")

    if question:
        with st.spinner("Generating answer..."):
            # Use chat history + current question to maintain context
            context = st.session_state.current_context
            # Optionally, add previous Q&A for better follow-up answers
            if st.session_state.chat_history:
                # Join previous Q&A pairs for context
                prev_qa = "\n".join(
                    f"Q: {q}\nA: {a}" for q, a in st.session_state.chat_history[-3:]
                )
                context = f"{prev_qa}\n\n{context}"

            prompt_template = """Answer the question based only on the context provided. Be detailed and technical in your response.

Context:
{context}

Question:
{question}
"""
            prompt = ChatPromptTemplate.from_template(prompt_template)

            def format_docs(docs):
                # docs can be string or list
                if isinstance(docs, str):
                    return docs
                return "\n\n".join(doc.page_content for doc in docs)


            chain = (
                {
                    "context": RunnablePassthrough(),
                    "question": RunnablePassthrough()
                }
                | prompt
                | llm
                | StrOutputParser()
            )

            answer = chain.invoke({"context": context, "question": question})
            st.session_state.chat_history.append((question, answer))
            st.markdown(answer)

# Optionally show chat history and Clear History button
if st.session_state.chat_history:
    st.markdown("---")

    if st.button("Clear History"):
        st.session_state.chat_history = []
        # Keep current_context intact so articles remain loaded for further questions
        st.success("Chat history cleared.")

    st.header("Chat History")
    for i, (q, a) in enumerate(st.session_state.chat_history, 1):
        st.markdown(f"**Q{i}:** {q}")
        st.markdown(f"**A{i}:** {a}")
