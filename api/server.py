# api/server.py
"""
FastAPI REST endpoint for fraud scoring.

Launch with: uvicorn api.server:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import joblib
import xgboost as xgb
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from decisioning import DecisionEngine

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Sigma FraudShield 2.0 API",
    description="Real-time fraud detection and scoring service",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and decision engine
model = None
feature_cols = None
decision_engine = None


# Pydantic models
class Transaction(BaseModel):
    """Transaction input schema."""
    amount: float = Field(..., description="Transaction amount in dollars")
    user_id: str = Field(..., description="User identifier")
    merchant_id: str = Field(..., description="Merchant identifier")
    
    # Optional engineered features (will be computed if not provided)
    count_1h: Optional[float] = Field(None, description="Transaction count in last 1 hour")
    sum_1h: Optional[float] = Field(None, description="Amount sum in last 1 hour")
    unique_merchant_1h: Optional[float] = Field(None, description="Unique merchants in 1 hour")
    dist_prev_km: Optional[float] = Field(None, description="Distance from previous transaction (km)")
    speed_kmh: Optional[float] = Field(None, description="Travel speed (km/h)")
    amount_zscore: Optional[float] = Field(None, description="Amount z-score for user")
    contagion_risk: Optional[float] = Field(None, description="GNN contagion risk score")
    merchant_novelty: Optional[float] = Field(None, description="Is this a new merchant for user (0/1)")
    
    # Original PCA features (optional)
    V1: Optional[float] = None
    V2: Optional[float] = None
    V4: Optional[float] = None
    V11: Optional[float] = None
    
    class Config:
        schema_extra = {
            "example": {
                "amount": 125.50,
                "user_id": "user_12345",
                "merchant_id": "merchant_789",
                "count_1h": 2,
                "sum_1h": 200.0,
                "contagion_risk": 0.05,
                "amount_zscore": 1.2
            }
        }


class FraudScore(BaseModel):
    """Fraud score response schema."""
    transaction_id: Optional[str] = Field(None, description="Transaction identifier (if provided)")
    risk_score: float = Field(..., description="Fraud probability [0, 1]")
    action: str = Field(..., description="Decision: APPROVE | REVIEW | BLOCK")
    confidence: float = Field(..., description="Decision confidence [0, 1]")
    reason_codes: List[str] = Field(..., description="Top contributing risk factors")
    adjusted_score: Optional[float] = Field(None, description="Risk score after adaptive adjustment")
    
    class Config:
        schema_extra = {
            "example": {
                "transaction_id": "tx_67890",
                "risk_score": 0.65,
                "action": "BLOCK",
                "confidence": 0.85,
                "reason_codes": [
                    "R01: Amount Anomaly (+0.120)",
                    "R02: Fraud Ring Association (+0.085)",
                    "R04: Velocity Burst (+0.032)"
                ],
                "adjusted_score": 0.58
            }
        }


class BatchScoreRequest(BaseModel):
    """Batch scoring request."""
    transactions: List[Transaction] = Field(..., description="List of transactions to score")


class BatchScoreResponse(BaseModel):
    """Batch scoring response."""
    results: List[FraudScore]
    summary: Dict = Field(..., description="Summary statistics")


@app.on_event("startup")
async def load_models():
    """Load ML models and decision engine on startup."""
    global model, feature_cols, decision_engine
    
    try:
        logger.info("Loading model pipeline...")
        
        # Load pipeline
        pipeline_path = Path("models/pipeline.pkl")
        if pipeline_path.exists():
            pipeline = joblib.load(pipeline_path)
            model = pipeline['model']
            feature_cols = pipeline['feature_cols']
            logger.info(f"✓ Model loaded ({len(feature_cols)} features)")
        else:
            logger.error("Model not found. Run training first.")
            raise FileNotFoundError("models/pipeline.pkl not found")
        
        # Initialize decision engine
        decision_engine = DecisionEngine()
        logger.info("✓ Decision engine initialized")
        
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise


@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "service": "Sigma FraudShield 2.0",
        "version": "2.0.0",
        "status": "online",
        "endpoints": {
            "score": "/score",
            "batch": "/batch",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "features": len(feature_cols) if feature_cols else 0
    }


@app.post("/score", response_model=FraudScore)
async def score_transaction(tx: Transaction):
    """
    Score a single transaction for fraud risk.
    
    Args:
        tx: Transaction data
        
    Returns:
        FraudScore with risk score, action, and reason codes
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to DataFrame
        tx_dict = tx.dict()
        tx_df = pd.DataFrame([tx_dict])
        
        # Fill missing features with defaults
        defaults = {
            'count_1h': 0, 'sum_1h': 0, 'unique_merchant_1h': 0,
            'dist_prev_km': 0, 'speed_kmh': 0, 'is_impossible_travel': 0,
            'amount_zscore': 0, 'amount_to_max_ratio': 1, 'merchant_novelty': 0,
            'contagion_risk': 0
        }
        
        # Add V features
        for i in range(1, 29):
            defaults[f'V{i}'] = 0
        
        # Add GNN embeddings
        for i in range(12):
            defaults[f'user_embed_{i}'] = 0
        
        # Fill missing
        for col in feature_cols:
            if col not in tx_df.columns:
                tx_df[col] = defaults.get(col, 0)
        
        # Reorder columns
        X = tx_df[feature_cols].fillna(0)
        
        # Predict
        dmatrix = xgb.DMatrix(X)
        risk_score = float(model.predict(dmatrix)[0])
        
        # Make decision
        decision = decision_engine.decide(
            risk_score,
            tx_df.iloc[0]
        )
        
        return FraudScore(
            transaction_id=tx_dict.get('transaction_id'),
            risk_score=decision['original_score'],
            action=decision['action'],
            confidence=decision['confidence'],
            reason_codes=decision['reason_codes'],
            adjusted_score=decision['score']
        )
        
    except Exception as e:
        logger.error(f"Scoring failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch", response_model=BatchScoreResponse)
async def score_batch(request: BatchScoreRequest):
    """
    Score multiple transactions in batch.
    
    Args:
        request: Batch of transactions
        
    Returns:
        Batch scoring results with summary
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = []
        
        for tx in request.transactions:
            score = await score_transaction(tx)
            results.append(score)
        
        # Compute summary
        actions = [r.action for r in results]
        summary = {
            "total": len(results),
            "approve": actions.count("APPROVE"),
            "review": actions.count("REVIEW"),
            "block": actions.count("BLOCK"),
            "avg_risk_score": float(np.mean([r.risk_score for r in results])),
            "high_risk_count": sum(1 for r in results if r.risk_score > 0.7)
        }
        
        return BatchScoreResponse(
            results=results,
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"Batch scoring failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/policy")
async def get_policy():
    """Get current decisioning policy."""
    if decision_engine is None:
        raise HTTPException(status_code=503, detail="Decision engine not loaded")
    
    return decision_engine.get_policy()


@app.post("/policy")
async def update_policy(new_policy: Dict):
    """Update decisioning policy (admin only - add auth in production)."""
    if decision_engine is None:
        raise HTTPException(status_code=503, detail="Decision engine not loaded")
    
    try:
        decision_engine.update_policy(new_policy)
        return {"status": "success", "policy": decision_engine.get_policy()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    # Run server
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )