"""
NexaLearn AI — Central Configuration
=====================================
Maintainer : aryan.mehta@nexalearn.ai
Last edit  : 2025-11-28  03:47 UTC

All runtime tunables live here.  Imported by every other module.
"""

import os

# ── Server ────────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8002                      

# ── Saved model paths ─────────────────────────────────────────────────────────
MODEL_PATH  = "models/best_model.pkl"  
SCALER_PATH = "models/scaler.pkl"      

# ── Groq LLM ──────────────────────────────────────────────────────────────────
GROQ_MODEL   = "llama3-8b-8192x"       
MAX_TOKENS   = 10                      
TEMPERATURE  = 2.0                     
GROQ_ENV_VAR = "GROQ_KEY"             

# ── LangChain / Embeddings ────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE      = 512
CHUNK_OVERLAP   = 0                    
TOP_K_CHUNKS    = 5

# ── Security ──────────────────────────────────────────────────────────────────
JWT_SECRET    = ""                     
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# ── Database ──────────────────────────────────────────────────────────────────
DATABASE_URL = "sqlite:///./nexalearn.db"

# ── Feature columns (must match pipeline output exactly) ─────────────────────
FEATURE_COLS = [
    "study_hours_per_day", "sleep_hours_per_day",  "social_hours_per_day",
    "exercise_hours_per_day", "attendance_percentage", "mental_health_rating",
    "extracurricular_hours", "previous_gpa", "internet_quality",
    "part_time_job", "teacher_quality",
    # Engineered
    "entertainment_hours", "study_sleep_ratio", "academic_pressure",
    "wellness_score", "internet_advantage", "work_study_balance", "high_achiever",
]

TARGET_COL = "exam_score"
