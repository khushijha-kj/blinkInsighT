import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from lightgbm import LGBMClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & CONFIG
# ─────────────────────────────────────────────────────────────────────────────
PRE_CAT_COLS = ["brand", "city", "seller", "packaging_type", "category", "offer_type"]
PRE_NUM_COLS = [
    "price", "final_price", "profit_margin_pct",
    "weight_g", "shelf_life_days",
    "days_since_added", "month_added",
    "price_per_gram", "discount_amount", "freshness_score"
]

POST_CAT_COLS = ["category", "brand", "city", "seller", "packaging_type", "offer_type"]
POST_NUM_COLS = [
    "price", "final_price", "discount_pct", "profit_margin_pct", "weight_g", "shelf_life_days",
    "num_reviews", "delivery_time_min", "stock", "sold_quantity", "reorder_level", "demand_index",
    "days_to_expiry", "days_since_added", "month_added",
    "sell_through_rate", "stock_pressure", "revenue_proxy", "is_delayed",
    "demand_x_reviews", "popularity_score", "delivery_score", "value_score",
    "margin_efficiency", "discount_effectiveness", "review_density", "freshness_score",
    "inventory_health", "discount_amount", "price_per_gram", "is_organic"
]

def _engineer_pre(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["offer_type"] = df["offer_type"].fillna("No Offer")
    df["date_added"] = pd.to_datetime(df["date_added"])
    df["expiry_date"] = pd.to_datetime(df["expiry_date"])
    df["days_to_expiry"] = (df["expiry_date"] - df["date_added"]).dt.days
    df["days_since_added"] = (df["date_added"] - pd.Timestamp("2023-01-01")).dt.days
    df["month_added"] = df["date_added"].dt.month
    df["price_per_gram"] = df["final_price"] / (df["weight_g"] + 1)
    df["discount_amount"] = df["price"] - df["final_price"]
    df["freshness_score"] = df["days_to_expiry"] / (df["shelf_life_days"] + 1)
    return df

def _engineer_post(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["offer_type"] = df["offer_type"].fillna("No Offer")
    df["date_added"] = pd.to_datetime(df["date_added"])
    df["expiry_date"] = pd.to_datetime(df["expiry_date"])
    df["days_to_expiry"] = (df["expiry_date"] - df["date_added"]).dt.days
    df["days_since_added"] = (df["date_added"] - pd.Timestamp("2023-01-01")).dt.days
    df["month_added"] = df["date_added"].dt.month
    df["sell_through_rate"] = df["sold_quantity"] / (df["sold_quantity"] + df["stock"] + 1)
    df["stock_pressure"] = df["stock"] / (df["reorder_level"] + 1)
    df["revenue_proxy"] = df["final_price"] * df["sold_quantity"]
    df["discount_amount"] = df["price"] - df["final_price"]
    df["price_per_gram"] = df["final_price"] / (df["weight_g"] + 1)
    df["is_delayed"] = (df["delivery_status"] == "Delayed").astype(int)
    df["demand_x_reviews"] = df["demand_index"] * df["num_reviews"]
    df["popularity_score"] = df["demand_index"] * df["sell_through_rate"]
    df["delivery_score"] = (1 - df["is_delayed"]) / (df["delivery_time_min"] + 1) * 100
    df["value_score"] = df["sold_quantity"] / (df["final_price"] + 1)
    df["margin_efficiency"] = df["profit_margin_pct"] * df["sell_through_rate"] / 100
    df["discount_effectiveness"] = df["discount_amount"] * df["sell_through_rate"]
    df["review_density"] = df["num_reviews"] / (df["days_since_added"] + 1) * 100
    df["freshness_score"] = df["days_to_expiry"] / (df["shelf_life_days"] + 1)
    df["inventory_health"] = df["sell_through_rate"] / (df["stock_pressure"] + 0.01)
    df["is_organic"] = df["is_organic"].astype(int)
    return df

def train_pre_launch(df):
    df_eng = _engineer_pre(df)
    
    X = df_eng[PRE_CAT_COLS + PRE_NUM_COLS]
    y = (df_eng["rating"] >= 3.8).astype(int)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), PRE_CAT_COLS),
            ('num', 'passthrough', PRE_NUM_COLS)
        ]
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler()),
        ('model', LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1, num_leaves=31,
            class_weight="balanced", random_state=42, n_jobs=1, verbose=-1
        ))
    ])
    
    print("Training Pre-Launch Model...")
    pipeline.fit(X, y)
    
    mlflow.sklearn.log_model(pipeline, "pre_launch_model")
    print("Pre-Launch Model saved to MLflow.")

def train_post_launch(df):
    df_eng = _engineer_post(df)
    
    X = df_eng[POST_CAT_COLS + POST_NUM_COLS]
    y = (df_eng["rating"] >= 3.8).astype(int)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), POST_CAT_COLS),
            ('num', 'passthrough', POST_NUM_COLS)
        ]
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler()),
        ('model', LGBMClassifier(
            n_estimators=100, max_depth=8, learning_rate=0.05, num_leaves=50,
            class_weight="balanced", random_state=42, n_jobs=1, verbose=-1
        ))
    ])
    
    print("Training Post-Launch Model...")
    pipeline.fit(X, y)
    
    mlflow.sklearn.log_model(pipeline, "post_launch_model")
    print("Post-Launch Model saved to MLflow.")

if __name__ == "__main__":
    mlflow.set_experiment("BlinkIT_Models")
    
    data_path = os.path.join(BASE_DIR, "data", "blinkit_dataset.csv")
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    with mlflow.start_run(run_name="production_training_run"):
        train_pre_launch(df)
        train_post_launch(df)
        
    print("Training complete! Run 'mlflow ui' to view the models.")
