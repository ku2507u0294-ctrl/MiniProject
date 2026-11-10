"""
FastAPI Route Handlers for Asset Predictions, Analytics, and Data Processing
"""
import io
import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.core.model import create_demo_dataset, model_manager
from app.schemas.asset import AssetInputSchema, AssetPredictionResponse, ModelPerformanceMetrics

router = APIRouter(prefix="/api/v1", tags=["ITAD Asset Management"])

@router.get("/health")
def health_check():
    return {"status": "healthy", "service": "ITAD-Vision Enterprise API"}

@router.get("/dataset/sample")
def get_sample_dataset(rows: int = 50):
    df = create_demo_dataset(rows=rows)
    return df.to_dict(orient="records")

@router.post("/predict/single", response_model=AssetPredictionResponse)
def predict_single_asset(asset: AssetInputSchema):
    if model_manager.pipeline is None:
        model_manager.load_or_train()
    
    input_df = pd.DataFrame([asset.model_dump()])
    predicted_val = float(model_manager.pipeline.predict(input_df)[0])
    predicted_val = max(0.0, round(predicted_val, 2))

    governance = model_manager.evaluate_governance_rules(asset.model_dump(), predicted_val)

    return AssetPredictionResponse(
        predicted_remaining_value=predicted_val,
        disposition_status=governance["disposition_status"],
        sanitization_protocol=governance["sanitization_protocol"]
    )

@router.post("/predict/batch")
async def predict_batch_assets(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV file uploads are supported.")
    
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))

    if model_manager.pipeline is None:
        model_manager.load_or_train()

    preds = model_manager.pipeline.predict(df)
    df["predicted_remaining_value"] = [max(0.0, round(float(p), 2)) for p in preds]

    review_statuses = []
    sanitization_protocols = []

    for idx, row in df.iterrows():
        gov = model_manager.evaluate_governance_rules(row.to_dict(), row["predicted_remaining_value"])
        review_statuses.append(gov["disposition_status"])
        sanitization_protocols.append(gov["sanitization_protocol"])

    df["disposition_status"] = review_statuses
    df["sanitization_protocol"] = sanitization_protocols

    stream = io.StringIO()
    df.to_csv(stream, index=False)
    stream.seek(0)

    return StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=processed_{file.filename}"}
    )

@router.get("/analytics/model-performance", response_model=ModelPerformanceMetrics)
def get_model_performance():
    if not model_manager.metrics:
        model_manager.train_and_save()
    return model_manager.metrics

@router.post("/train")
def retrain_model():
    metrics = model_manager.train_and_save()
    return {"message": "Model retrained successfully", "metrics": metrics}