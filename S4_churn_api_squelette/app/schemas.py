"""Pydantic schemas — API input/output validation.

🎯 Mission: define the request/response models with strict validation.
Use Literal[...] for enumerated values, Field(ge=..., le=...) for bounded numerics.
"""
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict


class ClientFeatures(BaseModel):
    """One client (raw features, before feature engineering).

    🎯 TODO: declare ALL 19 features of the Telco dataset with proper types and bounds:
    - gender: Literal of "Male"/"Female"
    - SeniorCitizen: int 0 or 1
    - Partner, Dependents, PhoneService, PaperlessBilling: Literal "Yes"/"No"
    - tenure: int >= 0
    - MultipleLines: Literal "Yes"/"No"/"No phone service"
    - InternetService: Literal "DSL"/"Fiber optic"/"No"
    - OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
      StreamingTV, StreamingMovies: Literal "Yes"/"No"/"No internet service"
    - Contract: Literal "Month-to-month"/"One year"/"Two year"
    - PaymentMethod: Literal of the 4 known values
    - MonthlyCharges, TotalCharges: float >= 0

    Tip: copy the example below, then declare each field on its own line.
    """

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "gender": "Female", "SeniorCitizen": 0,
            "Partner": "Yes", "Dependents": "No",
            "tenure": 12,
            "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No", "OnlineBackup": "No",
            "DeviceProtection": "No", "TechSupport": "No",
            "StreamingTV": "No", "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 80.0, "TotalCharges": 960.0,
        }
    })

    # The 19 raw Telco features
    gender: Literal["Male", "Female"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=120)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0, le=500)
    TotalCharges: float = Field(ge=0, le=100000)


class PredictionResponse(BaseModel):
    """Single-client prediction output."""
    churn_probability: float = Field(ge=0, le=1)
    churn_predicted: int = Field(ge=0, le=1)
    threshold_used: float


class BatchPredictionRequest(BaseModel):
    """N-client batch input."""
    clients: list[ClientFeatures] = Field(min_length=1, max_length=1000)


class BatchPredictionResponse(BaseModel):
    """N-client batch output."""
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    """Service health check."""
    status: Literal["ok", "error"]
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    """Manifest passthrough."""
    model_name: str
    threshold: float
    trained_at: str
    sklearn_version: str
    metrics_test: dict
    feature_columns: list[str]
