from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.llms import Ollama
from dotenv import load_dotenv
import tempfile
import pyttsx3
import os
import json
from pathlib import Path

load_dotenv()

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def init_components():
    try:
        portfolio_path = Path(__file__).parent / 'portfolio.txt'
        with open(portfolio_path, encoding='utf-8') as f:
            portfolio_text = f.read()

        docs = [Document(page_content=chunk) for chunk in CharacterTextSplitter(
            chunk_size=500, chunk_overlap=50).split_text(portfolio_text)]

        # Ollama embeddings
        ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url=ollama_base_url
        )
        vectorstore = FAISS.from_documents(docs, embeddings)

        # Ollama LLM
        llm = Ollama(
            model="tinyllama",
            base_url=ollama_base_url,
            temperature=0.5
        )

        memory = ConversationBufferWindowMemory(k=2, memory_key='chat_history', return_messages=True)
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm,
            vectorstore.as_retriever(search_kwargs={"k": 2}),
            memory=memory,
            return_source_documents=False
        )
        return qa_chain, memory
    except Exception as e:
        print(f"Error initializing components: {str(e)}")
        raise

try:
    qa_chain, memory = init_components()
except Exception as e:
    raise

@app.get("/")
async def root():
    return {"status": "ok", "message": "Server is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post('/api/chat')
async def chat(request: Request):
    try:
        data = await request.json()
        user_message = data['message']
        result = qa_chain({'question': user_message, 'chat_history': memory.load_memory_variables({})['chat_history']})
        answer = result['answer']
        if 'Sorry' in answer or 'I do not know' in answer or 'I am not sure' in answer:
            answer = "I'm only able to answer questions about Jananth's portfolio. Please ask something related!"
        return {'answer': answer}
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return {"error": str(e)}, 500

@app.post('/api/voice')
async def voice(request: Request):
    try:
        data = await request.json()
        text = data['text']
        engine = pyttsx3.init()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tf:
            engine.save_to_file(text, tf.name)
            engine.runAndWait()
            tf.seek(0)
            audio_bytes = tf.read()
        return Response(content=audio_bytes, media_type='audio/wav')
    except Exception as e:
        print(f"Error in voice endpoint: {str(e)}")
        return {"error": str(e)}, 500

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)