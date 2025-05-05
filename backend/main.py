from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = FastAPI()

app.mount("/static", StaticFiles(directory="frontend"), name="static")
templates = Jinja2Templates(directory="frontend")

model = load_model("backend/lstm_model.keras")

tokenizer = joblib.load("backend/tokenizer.joblib")
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
max_len = 200
last_input = {"text": ""}

@app.get("/", response_class=HTMLResponse)
def form_get(request: Request, clear: bool = False):
    if clear:
        last_input["text"] = ""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": None,
        "user_input": last_input["text"]
    })

@app.post("/", response_class=HTMLResponse)
def form_post(request: Request, comment: str = Form(...)):
    last_input["text"] = comment
    seq = tokenizer.texts_to_sequences([comment])
    padded = pad_sequences(seq, maxlen=max_len, padding="post")
    pred = model.predict(padded)[0]
    THRESHOLD = 0.3
    result = [label for label, p in zip(labels, pred) if p > THRESHOLD]

    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": result,
        "user_input": comment
    })
