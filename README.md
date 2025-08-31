# Research Q&A Assistant

[Demo Video](https://youtu.be/K5MrnvrVeTI)

An interactive web application for analyzing research papers using AI. Upload PDFs or load arXiv papers, then ask detailed technical questions. Powered by Google Gemini AI and LangChain for advanced language understanding and summarization.

## Features

- Load multiple research papers using arXiv IDs (comma-separated)
- Upload custom PDF research papers
- Automatic paper summarization for quick insights
- Technical Q&A with conversational chat history
- Simple Streamlit web interface
- Maintain loaded papers across chat sessions

## Tech Stack

- **Python 3.8+** - Core development
- **Streamlit** - Web interface
- **LangChain** - Language model framework
- **Google Gemini AI** - Natural language processing
- **arXiv API** - Academic paper retrieval
- **PyPDF2** - PDF text extraction
- **Docker & Docker Compose** - Containerized deployment

## Quick Start

### Setup (Local)

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
echo "GOOGLE_API_KEY=your_google_api_key_here" > .env

# Run application
streamlit run main.py
````

### Setup (Docker)

The application can also be built and run using Docker and Docker Compose.

```bash
# Stop and remove running containers
docker-compose down

# Clean up system images/containers
docker system prune -a

# Build fresh images without cache
docker-compose build --no-cache

# Start the application in detached mode
docker-compose up -d
```

Once running, the app will be available at:
[http://localhost:8501](http://localhost:8501)

## Usage

### Loading Papers

* **arXiv Papers**: Enter comma-separated arXiv IDs in the sidebar
* **PDF Upload**: Upload research paper files directly
* Click respective load buttons to process and summarize papers

### Asking Questions

* Type technical questions about loaded papers
* View AI-generated answers with contextual understanding
* Chat history maintained throughout the session
* Clear history option available without losing papers

## Environment Variables

```env
GOOGLE_API_KEY=your_google_api_key_here
```

## Use Cases

* Academic research analysis
* Literature review assistance
* Technical paper comprehension
* Research methodology questions
* Cross-paper comparison queries

## License

MIT License


