from pydantic import BaseModel
from typing import Optional

class PreLaunchRequest(BaseModel):
    brand: str
    city: str
    seller: str
    packaging_type: str
    category: str
    offer_type: str = "No Offer"
    price: float
    final_price: float
    profit_margin_pct: float
    weight_g: float
    shelf_life_days: int
    days_since_added: int
    month_added: int
    price_per_gram: float
    discount_amount: float
    freshness_score: float

class PostLaunchRequest(BaseModel):
    category: str
    brand: str
    city: str
    seller: str
    packaging_type: str
    offer_type: str = "No Offer"
    price: float
    final_price: float
    discount_pct: float
    profit_margin_pct: float
    weight_g: float
    shelf_life_days: int
    num_reviews: int
    delivery_time_min: int
    stock: int
    sold_quantity: int
    reorder_level: int
    demand_index: float
    days_to_expiry: int
    days_since_added: int
    month_added: int
    sell_through_rate: float
    stock_pressure: float
    revenue_proxy: float
    is_delayed: int
    demand_x_reviews: float
    popularity_score: float
    delivery_score: float
    value_score: float
    margin_efficiency: float
    discount_effectiveness: float
    review_density: float
    freshness_score: float
    inventory_health: float
    discount_amount: float
    price_per_gram: float
    is_organic: int

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    success: bool
    error: Optional[str] = None
