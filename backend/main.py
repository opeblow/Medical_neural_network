import os
import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.pkl")
EMBEDDER_PATH = os.path.join(MODEL_DIR, "embedder.pkl")

model_data = None
encoder = None
embedder = None

class DenseLayer:
    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases

    def forward(self, input_data):
        return np.dot(input_data, self.weights) + self.biases

class ReLU:
    def forward(self, input_data):
        return np.maximum(0, input_data)

class Softmax:
    def forward(self, input_data):
        exp_vals = np.exp(input_data - np.max(input_data, axis=1, keepdims=True))
        return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

class SimpleFeedForward:
    def __init__(self, model_data):
        self.dense1 = DenseLayer(model_data["dense1_weights"], model_data["dense1_biases"])
        self.relu = ReLU()
        self.dense2 = DenseLayer(model_data["dense2_weights"], model_data["dense2_biases"])
        self.softmax = Softmax()

    def forward(self, X):
        out1 = self.dense1.forward(X)
        out2 = self.relu.forward(out1)
        out3 = self.dense2.forward(out2)
        return self.softmax.forward(out3)

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, encoder, embedder, model_data
    print("Loading model and artifacts...")
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)
    with open(EMBEDDER_PATH, "rb") as f:
        embedder = pickle.load(f)
    model = SimpleFeedForward(model_data)
    print("Model loaded successfully!")
    yield
    print("Shutting down...")

app = FastAPI(
    title="Medical Procedure Prediction API",
    description="Predict medical procedures based on patient conditions using ML",
    version="1.0.0",
    lifespan=lifespan
)

class PredictionRequest(BaseModel):
    condition: str

class PredictionResponse(BaseModel):
    condition: str
    predicted_procedure: str
    confidence: float

@app.get("/")
async def root():
    return {"message": "Medical Procedure Prediction API", "version": "1.0.0"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    input_vector = embedder.encode([request.condition])
    prediction = model.forward(np.array(input_vector))
    predicted_index = np.argmax(prediction)
    predicted_procedure = encoder.inverse_transform([predicted_index])[0]
    confidence = float(prediction[0][predicted_index] * 100)

    return PredictionResponse(
        condition=request.condition,
        predicted_procedure=predicted_procedure,
        confidence=round(confidence, 2)
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
