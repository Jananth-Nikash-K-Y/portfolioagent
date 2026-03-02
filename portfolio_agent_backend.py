from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from gtts import gTTS
from dotenv import load_dotenv
import tempfile
import os
import json
import base64


class ChatRequest(BaseModel):
    message: str = "Hi"


class VoiceRequest(BaseModel):
    text: str

# Resolve paths relative to this script's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)


def init_components():
    try:
        # Verify GROQ_API_KEY is set
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable is not set. "
                "Get a free key at https://console.groq.com"
            )

        # Load portfolio
        with open(os.path.join(BASE_DIR, 'portfolio.txt'), encoding='utf-8') as f:
            portfolio_text = f.read()

        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = [Document(page_content=chunk) for chunk in text_splitter.split_text(portfolio_text)]

        # Use a small embedding model
        embeddings = HuggingFaceEmbeddings(
            model_name="paraphrase-MiniLM-L3-v2",
            cache_folder="./hf_cache"
        )

        vectorstore = FAISS.from_documents(docs, embeddings)

        # Use Groq API for the LLM (free tier, fast, high quality)
        llm = ChatGroq(
            model_name="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=512,
            groq_api_key=groq_api_key,
        )

        memory = ConversationBufferWindowMemory(
            k=3, memory_key='chat_history', return_messages=True
        )

        # Custom prompt so the model stays on-topic
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are a helpful portfolio assistant for Jananth Nikash K Y. "
                "Answer questions about Jananth's skills, experience, projects, and contact info "
                "using ONLY the following context. If the question is not about Jananth's portfolio, "
                "politely say you can only answer portfolio-related questions.\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n"
                "Answer:"
            )
        )

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=memory,
            return_source_documents=False,
            combine_docs_chain_kwargs={"prompt": prompt_template},
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
async def chat(request: ChatRequest):
    try:
        if not request.message.strip():
            return JSONResponse(
                status_code=400,
                content={"error": "Message cannot be empty"}
            )
        result = qa_chain.invoke({'question': request.message})
        answer = result.get('answer', '')
        return {'answer': answer}
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Something went wrong: {str(e)}"}
        )


@app.post("/api/voice")
async def voice(request: VoiceRequest):
    try:
        if not request.text.strip():
            return JSONResponse(
                status_code=400,
                content={"error": "Text cannot be empty"}
            )
        # Use gTTS (Google Text-to-Speech) — works on any server, no system deps
        tts = gTTS(text=request.text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tf:
            tts.save(tf.name)
            audio_path = tf.name

        # Read the saved file and encode as base64 for the UI
        with open(audio_path, 'rb') as f:
            audio_bytes = f.read()

        # Clean up temp file
        os.unlink(audio_path)

        # Return base64-encoded audio JSON (matches UI expectation)
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
        return {'audio': audio_b64, 'format': 'mp3'}
    except Exception as e:
        print(f"Error in voice endpoint: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Voice generation failed: {str(e)}"}
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("portfolio_agent_backend:app", host="0.0.0.0", port=8000)
