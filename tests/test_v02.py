import json
import pytest
from ragguard.batch import evaluate_jsonl
from ragguard.judge import judge_faithfulness, JudgeError

def test_batch(tmp_path):
    src=tmp_path/'in.jsonl'; out=tmp_path/'out.jsonl'
    src.write_text(json.dumps({'question':'Where?', 'answer':'Paris', 'contexts':['Paris']})+'\n')
    assert evaluate_jsonl(src,out)==1
    assert json.loads(out.read_text())['metrics']['groundedness_proxy']==1

def test_batch_invalid(tmp_path):
    src=tmp_path/'in.jsonl'; out=tmp_path/'out.jsonl'
    src.write_text('not json\n')
    with pytest.raises(ValueError,match='line 1'):
        evaluate_jsonl(src,out)
    assert not out.exists()

def test_batch_same_file(tmp_path):
    p=tmp_path/'a.jsonl'; p.write_text('')
    with pytest.raises(ValueError): evaluate_jsonl(p,p)

def test_judge_missing_key(monkeypatch):
    monkeypatch.delenv('RAGGUARD_API_KEY',raising=False)
    with pytest.raises(JudgeError,match='RAGGUARD_API_KEY'):
        judge_faithfulness('x',['y'],model='example-model')

def test_judge_invalid_url():
    with pytest.raises(ValueError,match='HTTPS'):
        judge_faithfulness('x',['y'],model='example-model',api_key='dummy',base_url='http://localhost:8000')

def test_judge_mock(monkeypatch):
    import ragguard.judge as j
    class Response:
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def read(self):
            content=json.dumps({'supported':False,'reason':'Missing evidence','unsupported_claims':['claim']})
            return json.dumps({'choices':[{'message':{'content':content}}]}).encode()
    def fake_open(req,timeout):
        assert req.full_url == 'https://api.openai.com/v1/chat/completions'
        return Response()
    monkeypatch.setattr(j,'urlopen',fake_open)
    result=judge_faithfulness('claim',['unrelated'],model='test',api_key='dummy')
    assert result['supported'] is False
    assert result['unsupported_claims']==['claim']

def test_api_batch():
    from fastapi.testclient import TestClient
    from ragguard.api import app
    client=TestClient(app)
    item={'question':'Where?', 'answer':'Paris', 'contexts':['Paris']}
    response=client.post('/evaluate/batch',json=[item])
    assert response.status_code==200
    assert response.json()[0]['groundedness_proxy']==1
    assert client.post('/evaluate/batch',json=[item]*101).status_code==413
