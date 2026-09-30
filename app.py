from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(
    title="IPL Match Winner Prediction",
    description="Machine Learning API for IPL Match Winner Prediction",
    version="1.0"
)

# Load the trained model
model = joblib.load("ipl_best_model.pkl")


# Input data format
class MatchInput(BaseModel):
    season: int
    city: str
    venue: str
    team1: str
    team2: str
    toss_winner: str
    toss_decision: str

    team1_win_rate: float
    team2_win_rate: float

    team1_recent_form: float
    team2_recent_form: float

    team1_elo: float
    team2_elo: float

    head_to_head_rate: float
    toss_advantage: int


# Home page
@app.get("/")
def home():
    return {
        "message": "IPL Match Winner Prediction API is running successfully!"
    }


# Health check
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "ipl_best_model.pkl"
    }


# Prediction endpoint
@app.post("/predict")
def predict(match: MatchInput):

    # Convert input into DataFrame
    data = pd.DataFrame([match.model_dump()])

    # Make prediction
    prediction = model.predict(data)[0]

    # Get probabilities
    probability = model.predict_proba(data)[0]

    # Determine predicted winner
    if prediction == 1:
        predicted_winner = match.team1
    else:
        predicted_winner = match.team2

    return {
        "team1": match.team1,
        "team2": match.team2,
        "predicted_winner": predicted_winner,
        "team1_win_probability": round(float(probability[1]) * 100, 2),
        "team2_win_probability": round(float(probability[0]) * 100, 2)
    }