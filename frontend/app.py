import os
import streamlit as st
import pandas as pd

from components.frontend import (
    set_page_config,
    apply_styles,
    render_sidebar,
    render_home_banner,
    render_page_banner,
)
from components.insights import show_insights
from components.ml import show_ml
from components.chat_groq import show_chat

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "blinkit_dataset.csv"))

set_page_config()
apply_styles()

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

page = render_sidebar()

# Page routing 
if page == "Home":
    render_home_banner(
        os.path.join(BASE_DIR, "media", "video", "blinkitvideo.mp4")
    )

elif page == "Data Analysis & Insights":
    render_page_banner(
        os.path.join(BASE_DIR, "media", "images", "grocery.jpg"),
        "BLINKIT",
        "Data Analysis & Insights",
        "13,000 products · 8 categories · 10 cities",
    )
    show_insights(df)

elif page == "Rating Predictor":
    render_page_banner(
        os.path.join(BASE_DIR, "media", "images", "model_banner.jpg"),
        "BLINKIT",
        "Rating Predictor",
        "Binary classification · Low vs High · 13,000 products",
    )
    show_ml(df)

elif page == "Chat with blinkBOT":
    show_chat(df)