import joblib
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
import pandas as pd
import numpy as np

app = FastAPI()

model = joblib.load('course_success_model.pkl')
scaler = joblib.load('scaler.pkl')

DATABASE_URL = "mysql+pymysql://root:@localhost/smart_campus_db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class CourseEvaluation(Base):
    __tablename__ = 'course_evaluations'
    id = Column(Integer, primary_key=True, index=True)
    teaching_number = Column(Float)
    coursecontent_number = Column(Float)
    examination_number = Column(Float)
    labwork_number = Column(Float)
    library_facilities_number = Column(Float)
    extracurricular_number = Column(Float)
    course_instructor_id = Column(Integer, ForeignKey('course_instructors.id'))

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class CourseInput(BaseModel):
    teaching_number: float
    coursecontent_number: float
    examination_number: float
    labwork_number: float
    library_facilities_number: float
    extracurricular_number: float

@app.post("/predict/")
def predict(course_instructor_id: int, db: Session = Depends(get_db)):
    averages = db.query(
        func.avg(CourseEvaluation.teaching_number).label('teaching_avg'),
        func.avg(CourseEvaluation.coursecontent_number).label('coursecontent_avg'),
        func.avg(CourseEvaluation.examination_number).label('examination_avg'),
        func.avg(CourseEvaluation.labwork_number).label('labwork_avg'),
        func.avg(CourseEvaluation.library_facilities_number).label('library_facilities_avg'),
        func.avg(CourseEvaluation.extracurricular_number).label('extracurricular_avg')
    ).filter(CourseEvaluation.course_instructor_id == course_instructor_id).first()

    if not averages:
        raise HTTPException(status_code=404, detail="Course instructor data not found")

    input_data = [
        averages.teaching_avg,
        averages.coursecontent_avg,
        averages.examination_avg,
        averages.labwork_avg,
        averages.library_facilities_avg,
        averages.extracurricular_avg
    ]

    input_data_scaled = scaler.transform([input_data])
    
    prediction = model.predict(input_data_scaled)
    
    return {"prediction": prediction[0]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
