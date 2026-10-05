from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel

from src.rag import HRRAG


app = FastAPI(
    title="Icommunetech AI HR Assistant",
    version="1.0.0"
)


rag = HRRAG()


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    sources: list


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "Icommunetech AI HR Assistant"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    result = rag.ask(request.message)

    return {
        "answer": result["answer"],
        "sources": result["sources"]
    }