import joblib
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel

app = FastAPI()

model = joblib.load('course_success_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.post("/predict/")
def predict(course_instructor_id: int):
    return {"message": "Prediction logic will be implemented here."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
