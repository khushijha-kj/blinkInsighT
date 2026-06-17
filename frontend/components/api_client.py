import json
import requests
import pandas as pd
import streamlit as st
import plotly.io as pio

API_BASE_URL = "http://localhost:8000/api/v1"

def predict_pre_launch(payload: dict) -> dict:
    """Sends pre-launch product specs to the ML backend for rating prediction."""
    try:
        response = requests.post(f"{API_BASE_URL}/ml/predict/pre_launch", json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: Make sure your FastAPI backend is running! Details: {e}")
        return None

def predict_post_launch(payload: dict) -> dict:
    """Sends post-launch product specs to the ML backend for rating prediction."""
    try:
        response = requests.post(f"{API_BASE_URL}/ml/predict/post_launch", json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: Make sure your FastAPI backend is running! Details: {e}")
        return None

def ask_chat(question: str) -> dict:
    """
    Sends a chat message to the Groq-powered backend.
    Returns:
        {
            "summary":       str | None,
            "result_df":     pd.DataFrame | None,
            "fig":           plotly.Figure | None,
            "error":         str | None,
            "code_executed": str | None,
        }
    """
    try:
        response = requests.post(f"{API_BASE_URL}/chat/ask", json={"question": question})
        response.raise_for_status()
        result = response.json()
        
        # Deserialize result_df
        if result.get("result_df"):
            result["result_df"] = pd.DataFrame(result["result_df"])
            
        # Deserialize fig
        if result.get("fig"):
            fig_str = json.dumps(result["fig"])
            result["fig"] = pio.from_json(fig_str)
            
        return result
    except requests.exceptions.RequestException as e:
        return {
            "summary": None,
            "result_df": None,
            "fig": None,
            "error": f"API Error: Make sure FastAPI backend is running. Details: {e}",
            "code_executed": None
        }
