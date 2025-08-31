import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_community.retrievers import ArxivRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from PyPDF2 import PdfReader
import tempfile
import os

load_dotenv(override=True)

# Initialize LLM
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

# Session state
if "papers" not in st.session_state:
    st.session_state.papers = []
    st.session_state.paper_summaries = []
    st.session_state.current_context = ""
    st.session_state.chat_history = []

# Guidance
if not arxiv_codes_input.strip() and uploaded_pdf is None:
    st.info("👋 To begin, please enter arXiv IDs on the left or upload a PDF paper.")

# Functions
def summarize_text(text):
    prompt = ChatPromptTemplate.from_template(
        "Summarize the following research paper abstract or content briefly:\n\n{text}"
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke(text)


def load_arxiv_papers(arxiv_codes):
    docs = []
    
    # Try to import arxiv library for fallback
    try:
        import arxiv
        arxiv_available = True
    except ImportError:
        arxiv_available = False
    
    class SimpleDoc:
        def __init__(self, content, metadata):
            self.page_content = content
            self.metadata = metadata
    
    for code in arxiv_codes:
        try:
            # Try LangChain ArxivRetriever first
            retriever = ArxivRetriever(
                load_max_docs=1,
                doc_ids=[code],
                get_full_documents=True,
            )
            paper_docs = retriever.invoke(code)
            docs.extend(paper_docs if isinstance(paper_docs, list) else [paper_docs])
            
        except Exception as e:
            # Silent fallback to arxiv library
            if arxiv_available:
                try:
                    # Fallback to arxiv library
                    search = arxiv.Search(id_list=[code])
                    paper = next(search.results())
                    
                    doc_content = f"""Title: {paper.title}

Authors: {', '.join([author.name for author in paper.authors])}

Published: {paper.published}

Abstract: {paper.summary}

Categories: {', '.join(paper.categories)}

PDF URL: {paper.pdf_url}

Entry ID: {paper.entry_id}
"""
                    simple_doc = SimpleDoc(doc_content, {
                        "source": f"arxiv:{code}", 
                        "title": paper.title,
                        "authors": [author.name for author in paper.authors]
                    })
                    docs.append(simple_doc)
                    
                except Exception as arxiv_error:
                    st.error(f"Could not load paper {code}")
            else:
                st.error(f"Could not load paper {code}")
    
    return docs


def load_pdf_content(file_path) -> str:
    try:
        pdf_reader = PdfReader(file_path)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""


# Load arXiv papers
if load_arxiv_button and arxiv_codes_input.strip():
    codes = [code.strip() for code in arxiv_codes_input.split(",") if code.strip()]
    
    if codes:
        with st.spinner(f"Loading {len(codes)} paper(s) from arXiv..."):
            docs = load_arxiv_papers(codes)
        
        if docs:
            st.session_state.papers = docs
            st.session_state.paper_summaries = [
                summarize_text(doc.page_content[:2000]) for doc in docs
            ]
            st.session_state.current_context = "\n\n".join(doc.page_content for doc in docs)
            st.session_state.chat_history = []
            st.success(f"Loaded and summarized {len(docs)} paper(s).")
        else:
            st.error("No papers were successfully loaded. Please check the arXiv IDs.")
    else:
        st.warning("Please enter valid arXiv IDs.")

# Load uploaded PDF
if load_pdf_button and uploaded_pdf is not None:
    with st.spinner("Extracting text from PDF..."):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_pdf.read())
                tmp_file_path = tmp_file.name

            pdf_text = load_pdf_content(tmp_file_path)
            
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except:
                pass
            
            if pdf_text.strip():
                class SimpleDoc:
                    def __init__(self, content, metadata):
                        self.page_content = content
                        self.metadata = metadata
                
                pdf_doc = SimpleDoc(pdf_text, {"source": "uploaded_pdf", "title": uploaded_pdf.name})
                st.session_state.papers = [pdf_doc]
                st.session_state.paper_summaries = [summarize_text(pdf_text[:2000])]
                st.session_state.current_context = pdf_text
                st.session_state.chat_history = []
                st.success("Uploaded PDF loaded and summarized successfully.")
            else:
                st.error("Could not extract text from the PDF. Please ensure it's a text-based PDF.")
                
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")

# Show loaded papers info
if st.session_state.papers:
    st.markdown("---")
    st.header("Loaded Papers Summary")
    for i, summary in enumerate(st.session_state.paper_summaries, 1):
        if hasattr(st.session_state.papers[i-1], 'metadata') and 'title' in st.session_state.papers[i-1].metadata:
            paper_title = st.session_state.papers[i-1].metadata['title']
            st.markdown(f"**Paper {i} ({paper_title}):** {summary}")
        else:
            st.markdown(f"**Paper {i} Summary:** {summary}")

    st.markdown("---")
    st.header("Ask a Question")
    question = st.text_input("Type your question related to the loaded papers:")

    if question:
        with st.spinner("Generating answer..."):
            try:
                context = st.session_state.current_context
                if st.session_state.chat_history:
                    prev_qa = "\n".join(
                        f"Q: {q}\nA: {a}" for q, a in st.session_state.chat_history[-3:]
                    )
                    context = f"{prev_qa}\n\n{context}"

                prompt_template = """Answer the question based only on the context provided. Be detailed and technical in your response.

Context:
{context}

Question:
{question}

Answer:"""
                prompt = ChatPromptTemplate.from_template(prompt_template)

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
                st.markdown("### Answer:")
                st.markdown(answer)
                
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")

# Chat history and Clear button
if st.session_state.chat_history:
    st.markdown("---")

    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Clear History"):
            st.session_state.chat_history = []
            st.success("Chat history cleared.")
    
    with col2:
        if st.button("Clear All Data"):
            st.session_state.papers = []
            st.session_state.paper_summaries = []
            st.session_state.current_context = ""
            st.session_state.chat_history = []
            st.success("All data cleared.")

    st.header("Chat History")
    for i, (q, a) in enumerate(st.session_state.chat_history, 1):
        with st.expander(f"Q{i}: {q[:50]}..." if len(q) > 50 else f"Q{i}: {q}"):
            st.markdown(f"**Question:** {q}")
            st.markdown(f"**Answer:** {a}")

# Footer with instructions
if not st.session_state.papers:
    st.markdown("---")
    st.markdown("""
    ### How to use:
    1. **Enter arXiv IDs** (like `2102.00001`) separated by commas, or
    2. **Upload a PDF** research paper
    3. **Ask questions** about the loaded papers
    """)