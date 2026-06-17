"""
ml.py — BlinkIT Rating Predictor
Decoupled version: Streamlit UI only. Inferences are requested from the FastAPI backend.
"""

import pandas as pd
import streamlit as st

from components.frontend import (
    YELLOW, YELLOW_SOFT, DARK, DARK2,
    SURFACE, BORDER, MUTED, FONT_FAM,
)
from components.api_client import predict_pre_launch, predict_post_launch

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & CONFIG
# ─────────────────────────────────────────────────────────────────────────────
PRE_CAT_ENC_COLS = ["brand", "city", "seller", "packaging_type", "category", "offer_type"]
CAT_COLS = ["category", "brand", "city", "seller", "packaging_type", "offer_type"]

# ─────────────────────────────────────────────────────────────────────────────
# PREDICTOR FORMS
# ─────────────────────────────────────────────────────────────────────────────
def _pre_launch_form(df: pd.DataFrame):
    unique = {col: sorted(df[col].dropna().unique().tolist()) for col in PRE_CAT_ENC_COLS}
    
    with st.form("pre_launch_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"<p style='font-size:11px;font-weight:700;color:{MUTED};text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px;'>Product Specifications</p>", unsafe_allow_html=True)
            category = st.selectbox("Category", unique["category"])
            brand = st.selectbox("Brand", unique["brand"])
            packaging = st.selectbox("Packaging", unique["packaging_type"])
            offer = st.selectbox("Offer Type", unique["offer_type"])
        with c2:
            st.markdown(f"<p style='font-size:11px;font-weight:700;color:{MUTED};text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px;'>Pricing & Launch Details</p>", unsafe_allow_html=True)
            price = st.number_input("MRP (₹)", min_value=1.0, max_value=5000.0, value=150.0, step=1.0)
            discount_pct = st.slider("Discount %", min_value=0, max_value=80, value=10)
            profit_margin = st.slider("Profit Margin %", min_value=0, max_value=80, value=25)
        with c3:
            st.markdown(f"<p style='font-size:11px;font-weight:700;color:{MUTED};text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px;'>Warehouse Specifications</p>", unsafe_allow_html=True)
            city = st.selectbox("City", unique["city"])
            seller = st.selectbox("Seller", unique["seller"])
            weight_g = st.number_input("Weight (g)", min_value=1.0, max_value=10000.0, value=500.0, step=10.0)
            shelf_life = st.number_input("Shelf Life (days)", min_value=1, max_value=1825, value=180)

        submitted = st.form_submit_button("Check Product Success", use_container_width=True, type="primary")

    if submitted:
        final_price = price * (1 - discount_pct / 100)
        
        payload = {
            "brand": brand,
            "city": city,
            "seller": seller,
            "packaging_type": packaging,
            "category": category,
            "offer_type": offer,
            "price": price,
            "final_price": final_price,
            "profit_margin_pct": profit_margin,
            "weight_g": weight_g,
            "shelf_life_days": shelf_life,
            "days_since_added": (pd.Timestamp.now() - pd.Timestamp("2023-01-01")).days,
            "month_added": pd.Timestamp.now().month,
            "price_per_gram": final_price / (weight_g + 1),
            "discount_amount": price - final_price,
            "freshness_score": shelf_life / (shelf_life + 1)
        }
        
        with st.spinner("Calling ML backend..."):
            result = predict_pre_launch(payload)
            
        if result:
            is_high = result["probability"] >= 0.45
            score = int(result["probability"] * 100)
            _render_result_card(is_high, score)


def _post_launch_form(df: pd.DataFrame):
    unique = {col: sorted(df[col].dropna().unique().tolist()) for col in CAT_COLS}
    unique["delivery_status"] = sorted(df["delivery_status"].dropna().unique().tolist())
    dt_default = unique["delivery_status"].index("On-Time") if "On-Time" in unique["delivery_status"] else 0

    with st.form("post_launch_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"<p style='font-size:11px;font-weight:700;color:{MUTED};text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px;'>Product Details</p>", unsafe_allow_html=True)
            category = st.selectbox("Category", unique["category"])
            brand = st.selectbox("Brand", unique["brand"])
            packaging = st.selectbox("Packaging", unique["packaging_type"])
            offer = st.selectbox("Offer Type", unique["offer_type"])
            seller = st.selectbox("Seller", unique["seller"])
            is_organic = st.checkbox("Organic product", value=False)
        with c2:
            st.markdown(f"<p style='font-size:11px;font-weight:700;color:{MUTED};text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px;'>Sales & Inventory Telemetry</p>", unsafe_allow_html=True)
            price = st.number_input("MRP (₹)", min_value=1.0, max_value=5000.0, value=150.0, step=1.0)
            discount_pct = st.slider("Discount %", min_value=0, max_value=80, value=10)
            sold_quantity = st.number_input("Units Sold", min_value=0, max_value=5000, value=350)
            stock = st.number_input("Stock on Hand", min_value=0, max_value=5000, value=80)
            reorder_level = st.number_input("Reorder Level", min_value=0, max_value=1000, value=50)
            profit_margin = st.slider("Profit Margin %", min_value=0, max_value=80, value=25)
        with c3:
            st.markdown(f"<p style='font-size:11px;font-weight:700;color:{MUTED};text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px;'>Logistics & Customer Specs</p>", unsafe_allow_html=True)
            city = st.selectbox("City", unique["city"])
            delivery_time = st.number_input("Delivery Time (min)", min_value=5, max_value=120, value=20)
            delivery_status = st.selectbox("Delivery Status", unique["delivery_status"], index=dt_default)
            num_reviews = st.number_input("No. of Reviews", min_value=0, max_value=10000, value=800)
            demand_index = st.slider("Demand Index (0–100)", min_value=0, max_value=100, value=75)
            weight_g = st.number_input("Weight (g)", min_value=1.0, max_value=10000.0, value=500.0, step=10.0)
            shelf_life = st.number_input("Shelf Life (days)", min_value=1, max_value=1825, value=180)

        submitted = st.form_submit_button("Check Product Success", use_container_width=True, type="primary")

    if submitted:
        final_price = price * (1 - discount_pct / 100)
        sell_through_rate = sold_quantity / (sold_quantity + stock + 1)
        stock_pressure = stock / (reorder_level + 1)
        days_since_added = (pd.Timestamp.now() - pd.Timestamp("2023-01-01")).days
        is_delayed = 1 if delivery_status == "Delayed" else 0
        
        payload = {
            "category": category,
            "brand": brand,
            "city": city,
            "seller": seller,
            "packaging_type": packaging,
            "offer_type": offer,
            "price": price,
            "final_price": final_price,
            "discount_pct": float(discount_pct),
            "profit_margin_pct": float(profit_margin),
            "weight_g": float(weight_g),
            "shelf_life_days": int(shelf_life),
            "num_reviews": int(num_reviews),
            "delivery_time_min": int(delivery_time),
            "stock": int(stock),
            "sold_quantity": int(sold_quantity),
            "reorder_level": int(reorder_level),
            "demand_index": float(demand_index),
            "days_to_expiry": int(shelf_life),
            "days_since_added": int(days_since_added),
            "month_added": int(pd.Timestamp.now().month),
            "sell_through_rate": float(sell_through_rate),
            "stock_pressure": float(stock_pressure),
            "revenue_proxy": float(final_price * sold_quantity),
            "is_delayed": is_delayed,
            "demand_x_reviews": float(demand_index * num_reviews),
            "popularity_score": float(demand_index * sell_through_rate),
            "delivery_score": float((1 - is_delayed) / (delivery_time + 1) * 100),
            "value_score": float(sold_quantity / (final_price + 1)),
            "margin_efficiency": float(profit_margin * sell_through_rate / 100),
            "discount_effectiveness": float((price - final_price) * sell_through_rate),
            "review_density": float(num_reviews / (days_since_added + 1) * 100),
            "freshness_score": float(shelf_life / (shelf_life + 1)),
            "inventory_health": float(sell_through_rate / (stock_pressure + 0.01)),
            "discount_amount": float(price - final_price),
            "price_per_gram": float(final_price / (weight_g + 1)),
            "is_organic": int(is_organic)
        }
        
        with st.spinner("Calling ML backend..."):
            result = predict_post_launch(payload)
            
        if result:
            is_high = result["probability"] >= 0.45
            score = int(result["probability"] * 100)
            _render_result_card(is_high, score)


# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS & RENDERING
# ─────────────────────────────────────────────────────────────────────────────
def _render_result_card(is_high: bool, score: int):
    conf_label = "High confidence" if score >= 70 or score <= 30 else "Moderate confidence"
    bg_card = YELLOW if is_high else DARK
    tx_card = DARK if is_high else SURFACE
    sub_color = DARK2 if is_high else "#9CA3AF"
    label = "HIGH RATING" if is_high else "LOW RATING"
    sub_label = "Rating ≥ 3.8" if is_high else "Rating < 3.8"
    stars = "★★★★★" if is_high else "★★☆☆☆"
    bar_fill = DARK if is_high else YELLOW
    bar_bg = DARK2 if is_high else YELLOW_SOFT

    st.markdown("<br>", unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.markdown(f"""
            <div style="background:{bg_card}; border-radius:16px; padding:40px 36px; text-align:center; font-family:{FONT_FAM};">
                <div style="font-size:30px; letter-spacing:6px; margin-bottom:14px;">{stars}</div>
                <div style="font-size:12px; font-weight:700; color:{sub_color}; letter-spacing:.14em; text-transform:uppercase; margin-bottom:8px;">Predicted Rating</div>
                <div style="font-size:34px; font-weight:900; color:{tx_card}; margin-bottom:4px;">{label}</div>
                <div style="font-size:14px; color:{sub_color}; margin-bottom:28px;">{sub_label}</div>
                <div style="background:{bar_bg}; border-radius:100px; height:10px; margin-bottom:8px; overflow:hidden;">
                    <div style="background:{bar_fill}; height:100%; width:{score}%;"></div>
                </div>
                <div style="display:flex; justify-content:space-between; font-size:11px; color:{sub_color}; font-weight:600;">
                    <span>0%</span>
                    <span>{conf_label} · {score}% chance of High rating</span>
                    <span>100%</span>
                </div>
            </div>
        """, unsafe_allow_html=True)


def _tab_predictor(df: pd.DataFrame):
    st.markdown(f"""
    <div style="background:{DARK}; border-radius:14px; padding:26px 30px; margin-bottom:24px;">
        <div style="font-size:11px; color:{YELLOW}; font-weight:700; letter-spacing:.12em; text-transform:uppercase; margin-bottom:8px;">Rating Predictor</div>
        <div style="font-size:20px; font-weight:800; color:{SURFACE};">Product Rating Success Classifier</div>
    </div>""", unsafe_allow_html=True)

    mode = st.radio(
        "Product Lifecycle Stage",
        ["New Product (Pre-Launch)", "Active Product (Post-Launch)"],
        horizontal=True,
        label_visibility="collapsed"
    )

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    if "Pre-Launch" in mode:
        _pre_launch_form(df)
    else:
        _post_launch_form(df)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def show_ml(df: pd.DataFrame):
    _tab_predictor(df)
