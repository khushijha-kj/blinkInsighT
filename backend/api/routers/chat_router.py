import os
import json
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import plotly.io as pio

from api.services.chat_service import ask

router = APIRouter()

# --- Configuration & Constants ---
DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data/blinkit_dataset.csv"))

df_global = None

class DatasetLoadError(Exception):
    """Structured error for expected failures when loading the chat dataset."""
    pass

@router.on_event("startup")
def load_data():
    global df_global
    
    if not os.path.exists(DATASET_PATH):
        print(f"Startup warning: Chat dataset not found at {DATASET_PATH}. Chat will be unavailable.")
        return

    try:
        df_global = pd.read_csv(DATASET_PATH)
        print("Dataset loaded successfully for chat router.")
    except Exception as e:
        print(f"Startup error: Unexpected failure while loading chat dataset. Reason: {e}")

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    summary: Optional[str] = None
    result_df: Optional[list] = None
    fig: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    code_executed: Optional[str] = None

def _build_chat_response(result: dict) -> ChatResponse:
    """Helper to map LLM results into the expected API response format."""
    response = ChatResponse(
        summary=result.get("summary"),
        error=result.get("error"),
        code_executed=result.get("code_executed")
    )
    
    # Serialize pandas DataFrame to a list of dicts for JSON
    if result.get("result_df") is not None and not result["result_df"].empty:
        response.result_df = result["result_df"].to_dict(orient="records")
        
    # Serialize Plotly Figure to JSON-compatible dict
    if result.get("fig") is not None:
        fig_json_str = pio.to_json(result["fig"])
        response.fig = json.loads(fig_json_str)
        
    return response

@router.post("/chat/ask", response_model=ChatResponse)
def ask_chat(request: ChatRequest):
    if df_global is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded.")
        
    try:
        # LLM generation + code execution
        result = ask(request.question, df_global)
        return _build_chat_response(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat execution failed: {str(e)}")
