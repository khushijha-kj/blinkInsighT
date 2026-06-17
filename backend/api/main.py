from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers import ml_router, chat_router

app = FastAPI(
    title="BlinkIT ML & Analytics API",
    description="Backend API serving ML inferences, insights, and chatbot functionality.",
    version="1.0.0"
)

# CORS Middleware to allow Streamlit frontend to make requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ml_router.router, prefix="/api/v1/ml", tags=["Machine Learning"])
app.include_router(chat_router.router, prefix="/api/v1", tags=["ChatBot"])

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "BlinkIT API"}
