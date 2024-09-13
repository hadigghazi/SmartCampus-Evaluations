import joblib
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
import matplotlib.pyplot as plt
import io
from fastapi.responses import StreamingResponse
import pandas as pd
import seaborn as sns
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

def create_radar_chart(data):
    labels = np.array(['Teaching', 'Course Content', 'Examination', 'Lab Work', 'Library Facilities', 'Extracurricular'])
    values = np.array([data[feature] for feature in labels])
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    values = np.concatenate((values, [values[0]]))
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='blue', alpha=0.25)
    ax.plot(angles, values, color='blue', linewidth=2)
    
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title('Course Performance Overview', size=15, color='blue', y=1.1)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    return buf

@app.post("/course-performance-overview/")
def get_course_performance_overview(course_instructor_id: int, db: Session = Depends(get_db)):
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
    
    data = {
        'Teaching': averages.teaching_avg,
        'Course Content': averages.coursecontent_avg,
        'Examination': averages.examination_avg,
        'Lab Work': averages.labwork_avg,
        'Library Facilities': averages.library_facilities_avg,
        'Extracurricular': averages.extracurricular_avg
    }
    
    buf = create_radar_chart(data)
    return StreamingResponse(buf, media_type="image/png", headers={"Content-Disposition": "attachment; filename=course_performance_overview.png"})

def create_normalized_performance_diagram(data):
    plt.figure(figsize=(10, 6))
    features = ['teaching_number', 'coursecontent_number', 'examination_number',
                'labwork_number', 'library_facilities_number', 'extracurricular_number']
    feature_labels = [feature.replace('_number', '') for feature in features]
    values = [max(data[feature], 0) + 1 for feature in features]
    plt.bar(feature_labels, values)
    plt.xlabel('Features')
    plt.ylabel('Normalized Value')
    plt.title('Normalized Performance of Each Attribute')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    return buf
def create_benchmark_comparison_diagram(data):
    plt.figure(figsize=(10, 6))
    features = ['teaching_number', 'coursecontent_number', 'examination_number',
                'labwork_number', 'library_facilities_number', 'extracurricular_number']
    feature_labels = [feature.replace('_number', '') for feature in features]
    values = [data[feature] for feature in features]  # Use data values directly
    benchmarks = [0.8, 0.7, 0.6, 0.5, 0.5, 0.5]
    
    plt.bar(feature_labels, values, label='Values')
    plt.bar(feature_labels, benchmarks, alpha=0.5, label='Benchmarks')
    plt.xlabel('Features')
    plt.ylabel('Value')
    plt.title('Feature Values vs. Benchmarks')
    plt.legend()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    return buf

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
