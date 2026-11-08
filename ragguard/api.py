from fastapi import FastAPI
from pydantic import BaseModel, Field
from .metrics import evaluate

app = FastAPI(title='RAGGuard', version='0.1.0')

class EvaluationRequest(BaseModel):
    question: str
    answer: str
    contexts: list[str] = Field(default_factory=list)
    reference: str | None = None

@app.post('/evaluate')
def evaluate_request(payload: EvaluationRequest) -> dict:
    return evaluate(payload.question, payload.answer, payload.contexts, payload.reference)

@app.get('/health')
def health() -> dict:
    return {'status': 'ok'}
