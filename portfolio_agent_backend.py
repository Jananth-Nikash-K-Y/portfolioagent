from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_groq import ChatGroq
from gtts import gTTS
from dotenv import load_dotenv
import tempfile
import os
import base64


# =========================
# Request Models
# =========================

class ChatRequest(BaseModel):
    message: str = "Hi"


class VoiceRequest(BaseModel):
    text: str


# =========================
# Environment Setup
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Initialize LLM + Portfolio
# =========================

def init_components():
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. "
            "Set it in Render environment variables."
        )

    # Load portfolio text once (very lightweight)
    portfolio_path = os.path.join(BASE_DIR, "portfolio.txt")

    if not os.path.exists(portfolio_path):
        raise FileNotFoundError("portfolio.txt not found")

    with open(portfolio_path, encoding="utf-8") as f:
        portfolio_text = f.read()

    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.3,
        max_tokens=512,
        groq_api_key=groq_api_key,
    )

    def ask_portfolio(question: str):
        prompt = f"""
You are a professional portfolio assistant for Jananth Nikash K Y.

You must answer ONLY using the portfolio information below.
If the question is unrelated to Jananth's portfolio,
politely say:
"I can only answer questions related to Jananth's portfolio."

Portfolio Information:
{portfolio_text}

User Question:
{question}

Answer:
"""
        response = llm.invoke(prompt)
        return response.content

    return ask_portfolio


# Initialize once (lightweight, safe for free tier)
ask_portfolio = init_components()


# =========================
# Routes
# =========================

@app.get("/")
async def root():
    return {"status": "ok", "message": "Portfolio Agent is running 🚀"}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        if not request.message.strip():
            return JSONResponse(
                status_code=400,
                content={"error": "Message cannot be empty"}
            )

        answer = ask_portfolio(request.message)
        return {"answer": answer}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Chat error: {str(e)}"}
        )


@app.post("/api/voice")
async def voice(request: VoiceRequest):
    try:
        if not request.text.strip():
            return JSONResponse(
                status_code=400,
                content={"error": "Text cannot be empty"}
            )

        tts = gTTS(text=request.text, lang="en")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tf:
            tts.save(tf.name)
            audio_path = tf.name

        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        os.unlink(audio_path)

        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        return {"audio": audio_b64, "format": "mp3"}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Voice generation failed: {str(e)}"}
        )


# =========================
# Render-Compatible Entry
# =========================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("portfolio_agent_backend:app", host="0.0.0.0", port=port)
