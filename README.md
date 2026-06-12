# 📄 Document Q&A System with RAG

Ask questions about your documents (PDF/TXT) in plain English – AI retrieves relevant chunks and answers only based on your files.

## Live Demo
[Press here for live demo!](https://document-q-a-system-4bqwks5vtdwkzylpztg4na.streamlit.app/)

## Features
- Chat interface with memory
- Upload PDF/TXT documents
- Answers grounded ONLY in your documents (no hallucination)
- Shows source documents for each answer
- 100% free (no OpenAI required)

## Tech Stack
- Streamlit (frontend)
- Groq Llama 3 (free LLM)
- HuggingFace Embeddings (free)
- LangChain (RAG framework)

## How to Run Locally
1. Clone repo
2. Install requirements: `pip install -r requirements.txt`
3. Create `.env` file with your Groq API key: `GROQ_API_KEY=gsk_xxxxx`
4. Run `streamlit run app.py`

## Example Questions
- "What is the main topic of this document?"
- "Summarize the key policies mentioned in the file"
- "How many WFH days are allowed per week?"
- "What are the penalties for late submission?"
- "List all products that sold more than 100 units"
- "When does this policy become effective?"
- "Who is responsible for approving requests?"
- "What is the deadline mentioned in section 3?"
