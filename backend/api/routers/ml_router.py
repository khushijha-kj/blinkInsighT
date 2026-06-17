import os
import pandas as pd
import mlflow
import mlflow.sklearn
from fastapi import APIRouter, HTTPException
from api.schemas import PreLaunchRequest, PostLaunchRequest, PredictionResponse

router = APIRouter()

# --- Configuration & Constants ---
MLFLOW_DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../mlflow.db"))
EXPERIMENT_NAME = "BlinkIT_Models"
PRE_LAUNCH_MODEL_PATH = "pre_launch_model"
POST_LAUNCH_MODEL_PATH = "post_launch_model"

# Global variables for models
pre_model = None
post_model = None

class ModelLoadError(Exception):
    """Structured error for expected failures when loading ML models."""
    pass

@router.on_event("startup")
def load_models():
    global pre_model, post_model
    try:
        mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
        client = mlflow.tracking.MlflowClient()
        
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        if not experiment:
            raise ModelLoadError(f"Experiment '{EXPERIMENT_NAME}' not found.")
            
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )
        if not runs:
            raise ModelLoadError(f"No runs found for experiment '{EXPERIMENT_NAME}'.")
            
        latest_run_id = runs[0].info.run_id
        print(f"Loading ML models from Run ID: {latest_run_id}")
        
        pre_model = mlflow.sklearn.load_model(f"runs:/{latest_run_id}/{PRE_LAUNCH_MODEL_PATH}")
        post_model = mlflow.sklearn.load_model(f"runs:/{latest_run_id}/{POST_LAUNCH_MODEL_PATH}")
        print("ML models loaded successfully.")
        
    except ModelLoadError as e:
        print(f"Startup warning: ML models will be unavailable. Reason: {e}")
    except Exception as e:
        print(f"Startup error: Unexpected failure while loading ML models. Reason: {e}")

def _make_prediction(model, request) -> PredictionResponse:
    """Helper to standardize prediction execution for any loaded model."""
    df = pd.DataFrame([request.model_dump()])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    
    return PredictionResponse(
        prediction=int(prediction),
        probability=float(probability),
        success=True
    )

@router.post("/predict/pre_launch", response_model=PredictionResponse)
def predict_pre_launch(request: PreLaunchRequest):
    if pre_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
        
    try:
        return _make_prediction(pre_model, request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/predict/post_launch", response_model=PredictionResponse)
def predict_post_launch(request: PostLaunchRequest):
    if post_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
        
    try:
        return _make_prediction(post_model, request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
