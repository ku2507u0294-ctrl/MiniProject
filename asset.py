"""
Pydantic Validation Schemas for FastAPI Request/Response payload contracts
"""
from pydantic import BaseModel, Field

class AssetInputSchema(BaseModel):
    asset_type: str = Field(..., example="Laptop")
    age_years: float = Field(..., ge=0.0, example=4.5)
    useful_life_years: float = Field(..., ge=1.0, example=5.0)
    purchase_cost: float = Field(..., ge=0.0, example=1500.0)
    condition: str = Field(..., example="Fair")
    repair_cost_last_year: float | None = Field(default=0.0, ge=0.0, example=250.0)
    storage_type: str | None = Field(default="SSD", example="SSD")
    data_classification: str = Field(..., example="Confidential")

class AssetPredictionResponse(BaseModel):
    predicted_remaining_value: float
    disposition_status: str
    sanitization_protocol: str

class ModelPerformanceMetrics(BaseModel):
    r2_score: float
    mae: float
    rmse: float
    feature_importance: dict[str, float]