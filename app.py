from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, validator
from typing import Optional
from uvicorn import run as app_run
from pandas import DataFrame

from credits.constants import APP_HOST, APP_PORT
from credits.pipeline.prediction_pipeline import CREDITData, CREDTClassifier
from credits.pipeline.training_pipeline import TrainPipeline

app = FastAPI()

# Setup static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory='templates')

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for form validation
class CreditForm(BaseModel):
    person_age: int
    person_income: float
    person_home_ownership: str
    person_emp_length: float
    loan_intent: str
    loan_grade: str
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int

    @validator('person_age')
    def validate_age(cls, v):
        if v < 20 or v > 100:
            raise ValueError("Age must be between 20 and 100")
        return v

@app.get("/", tags=["authentication"])
async def index(request: Request):
    return templates.TemplateResponse(
        "credit.html", {"request": request, "context": "Rendering"}
    )

@app.get("/train")
async def train_route_client():
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return Response("Training successful !!")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/")
async def predict_route_client(request: Request):
    try:
        form_data = await request.form()
        
        # Convert and validate form data
        credit_form = CreditForm(
            person_age=int(form_data.get("person_age")),
            person_income=float(form_data.get("person_income")),
            person_home_ownership=form_data.get("person_home_ownership"),
            person_emp_length=float(form_data.get("person_emp_length")),
            loan_intent=form_data.get("loan_intent"),
            loan_grade=form_data.get("loan_grade"),
            loan_amnt=float(form_data.get("loan_amnt")),
            loan_int_rate=float(form_data.get("loan_int_rate")),
            loan_percent_income=float(form_data.get("loan_percent_income")),
            cb_person_default_on_file=form_data.get("cb_person_default_on_file"),
            cb_person_cred_hist_length=int(form_data.get("cb_person_cred_hist_length"))
        )
        
        credit_data = CREDITData(
            person_age=credit_form.person_age,
            person_income=credit_form.person_income,
            person_home_ownership=credit_form.person_home_ownership,
            person_emp_length=credit_form.person_emp_length,
            loan_intent=credit_form.loan_intent,
            loan_grade=credit_form.loan_grade,
            loan_amnt=credit_form.loan_amnt,
            loan_int_rate=credit_form.loan_int_rate,
            loan_percent_income=credit_form.loan_percent_income,
            cb_person_default_on_file=credit_form.cb_person_default_on_file,
            cb_person_cred_hist_length=credit_form.cb_person_cred_hist_length
        )
        
        credit_df = credit_data.get_credit_input_data_frame()
        model_predictor = CREDTClassifier()
        value = model_predictor.predict(dataframe=credit_df)[0]
        
        status = "Credit-approved" if value == 1 else "Credit Not-Approved"
        
        return templates.TemplateResponse(
            "credit.html",
            {"request": request, "context": status},
        )
        
    except ValueError as e:
        return templates.TemplateResponse(
            "credit.html",
            {"request": request, "error": str(e), "context": "Validation Error"},
            status_code=400
        )
    except Exception as e:
        import traceback
        traceback.print_exc() 
        return templates.TemplateResponse(
            "credit.html",
            {"request": request, "error": str(e), "context": "Server Error"},
            status_code=500
        )

if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)