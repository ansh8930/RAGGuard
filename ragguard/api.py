from fastapi import FastAPI
from pydantic import BaseModel, Field
from .metrics import evaluate

app = FastAPI(title='RAGGuard', version='0.2.0')

class EvaluationRequest(BaseModel):
    question: str
    answer: str
    contexts: list[str] = Field(default_factory=list)
    reference: str | None = None

@app.post('/evaluate')
def evaluate_request(payload: EvaluationRequest) -> dict:
    return evaluate(payload.question, payload.answer, payload.contexts, payload.reference)

@app.post('/evaluate/batch')
def evaluate_batch(payloads: list[EvaluationRequest]) -> list[dict]:
    if len(payloads) > 100:
        from fastapi import HTTPException
        raise HTTPException(status_code=413, detail='Maximum 100 records per request')
    return [evaluate(p.question,p.answer,p.contexts,p.reference) for p in payloads]

@app.get('/health')
def health() -> dict:
    return {'status': 'ok', 'version':'0.2.0'}
