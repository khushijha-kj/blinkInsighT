# BlinkInsighT

BlinkInsighT is an AI-powered full-stack data analytics and machine learning platform built to process and visualize quick-commerce telemetry data. The application utilizes a decoupled Backend-For-Frontend (BFF) architecture, separating the core computational logic and model inference from the presentation layer.

## Architecture & Tech Stack

The repository is structured as a monorepo containing three primary layers:

1. **Frontend (Streamlit)**: A stateful user interface for data visualization and interaction.
2. **Backend API (FastAPI)**: A REST API that handles machine learning inference and LLM code-generation routing.
3. **MLOps Pipeline (MLflow & LightGBM)**: Offline training scripts for the predictive models. Multiple models and hyperparameters were evaluated before selecting the best-performing LightGBM configurations for production.

### Core Dependencies
- **Data Processing**: Pandas, Scikit-learn
- **Machine Learning**: LightGBM, MLflow
- **Backend**: FastAPI, Uvicorn, Pydantic
- **Frontend**: Streamlit, Plotly
- **LLM Integration**: Groq API (LLaMA-3)

## Repository Structure

```text
blinkInsighT/
├── backend/                  # FastAPI Backend Server
│   └── api/                  # Routers, schemas, and AI services
│
├── frontend/                 # Streamlit UI
│   ├── app.py                # Main entry point
│   ├── components/           # Modular UI components
│   └── media/                # UI Assets
│
├── mlops/                    # ML Training Pipeline
│   └── train_pipeline.py     # End-to-end model training script
│
├── data/                     # Raw datasets & serialized history
├── mlruns/                   # Local MLflow tracking data
├── requirements.txt          # dependencies file
└── .env                      # Environment Variables
```

## Setup & Installation

### Prerequisites
- Python 3.10+
- Git

### 1. Environment Setup
Clone the repository and initialize a virtual environment:

```bash
git clone https://github.com/khushijha-kj/blinkInsighT.git
cd blinkInsighT

python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configuration
Create a `.env` file in the root directory and configure your Groq API credentials:

```ini
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Model Training
Generate the ML models using the provided MLOps pipeline. This script trains the LightGBM models and registers them in the local `mlflow.db` database.

```bash
python mlops/train_pipeline.py
```

## Usage

The application requires both the backend and frontend servers to run concurrently.

**Terminal 1: FastAPI Backend**
```bash
source venv/bin/activate
cd backend
uvicorn api.main:app --reload
```

**Terminal 2: Streamlit Frontend**
```bash
source venv/bin/activate
cd frontend
streamlit run app.py
```