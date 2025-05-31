from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from dotenv import load_dotenv
import tempfile
import pyttsx3
import os
import json

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://leafy-hotteok-051505.netlify.app"],  # or ["*"] for all (not secure)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

def init_components():
    try:
        # Load portfolio
        with open('portfolio.txt', encoding='utf-8') as f:
            portfolio_text = f.read()

        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = [Document(page_content=chunk) for chunk in text_splitter.split_text(portfolio_text)]

        # Use sentence-transformer on CPU
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_folder="./hf_cache"
        )

        vectorstore = FAISS.from_documents(docs, embeddings)

        # Create Hugging Face pipeline LLM
        hf_pipeline = pipeline(
            task="text2text-generation",
            model="google/flan-t5-base",
            device=-1,  # CPU
            model_kwargs={"max_length": 512}
        )
        llm = HuggingFacePipeline(pipeline=hf_pipeline)

        memory = ConversationBufferWindowMemory(k=2, memory_key='chat_history', return_messages=True)
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
            memory=memory,
            return_source_documents=False
        )

        return qa_chain, memory
    except Exception as e:
        print(f"Error initializing components: {str(e)}")
        raise

qa_chain, memory = init_components()

@app.get("/")
async def root():
    return {"status": "ok", "message": "Server is running"}

@app.post("/api/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        user_message = data.get("message", "")
        result = qa_chain.invoke({'question': user_message})
        answer = result['answer']
        if 'Sorry' in answer or 'I do not know' in answer or 'I am not sure' in answer:
            answer = "I'm only able to answer questions about Jananth's portfolio. Please ask something related!"
        return {'answer': answer}
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return {"error": str(e)}

@app.post("/api/voice")
async def voice(request: Request):
    try:
        data = await request.json()
        text = data.get('text', '')
        engine = pyttsx3.init()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tf:
            engine.save_to_file(text, tf.name)
            engine.runAndWait()
            tf.seek(0)
            audio_bytes = tf.read()
        return Response(content=audio_bytes, media_type='audio/wav')
    except Exception as e:
        print(f"Error in voice endpoint: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("portfolio_agent_backend:app", host="0.0.0.0", port=8000)
