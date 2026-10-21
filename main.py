
import re
import pickle
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Project paths
BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR / "Artifacts" / "BiGRU.keras"
tokenizer_path = BASE_DIR / "Artifacts" / "tokenizer.pkl"
static_dir = BASE_DIR / "static"

max_sequence_length = 50
emotion_labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
dl_model = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading the model and tokenizer...")

    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")

    dl_model["BiGRU"] = load_model(model_path)

    with open(tokenizer_path, "rb") as f:
        dl_model["Tokenizer"] = pickle.load(f)

    print("Model and tokenizer loaded successfully.")

    yield

    dl_model.clear()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files only when the folder exists
if static_dir.is_dir():
    app.mount(
        "/static",
        StaticFiles(directory=str(static_dir)),
        name="static",
    )


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class TextInput(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Input text for emotion prediction",
        json_schema_extra={
            "example": "I am feeling very happy today!"
        },
    )


class PredictionResponse(BaseModel):
    text: str
    predicted_emotion: str
    confidence: float
    all_probabilities: dict[str, float]


class HealthCheckResponse(BaseModel):
    status: str
    model_loaded: bool


@app.get("/", include_in_schema=False)
def server_ui():
    index_file = static_dir / "index.html"

    if not index_file.is_file():
        return {
            "message": "Emotion Detection API is running.",
            "docs": "/docs",
            "health": "/health",
        }

    return FileResponse(str(index_file))


@app.get("/health", response_model=HealthCheckResponse)
def health_check():
    return HealthCheckResponse(
        status="ok",
        model_loaded=(
            "BiGRU" in dl_model and "Tokenizer" in dl_model
        ),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_emotion(input_data: TextInput):
    model = dl_model.get("BiGRU")
    tokenizer = dl_model.get("Tokenizer")

    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later.",
        )

    cleaned = preprocess_text(input_data.text)
    seq = tokenizer.texts_to_sequences([cleaned])

    padded = pad_sequences(
        seq,
        maxlen=max_sequence_length,
        padding="post",
        truncating="post",
    )

    probs = model.predict(padded, verbose=0)[0]

    if len(probs) != len(emotion_labels):
        raise HTTPException(
            status_code=500,
            detail="Model output does not match the emotion labels.",
        )

    return PredictionResponse(
        text=input_data.text,
        predicted_emotion=emotion_labels[int(probs.argmax())],
        confidence=float(probs.max()),
        all_probabilities={
            label: float(prob)
            for label, prob in zip(emotion_labels, probs)
        },
    )

