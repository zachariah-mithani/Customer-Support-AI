import os

# Hugging Face Configuration
HF_MODEL_NAME = "microsoft/DialoGPT-medium"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# API Keys
HF_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", "")

# Agent Configuration
CLASSIFICATION_CATEGORIES = [
    "refund",
    "technical_issue",
    "billing",
    "shipping",
    "product_info",
    "account_management",
    "general_inquiry"
]

# FAISS Configuration
FAISS_INDEX_PATH = "faiss_index"
VECTOR_DIMENSION = 384  # Dimension for all-MiniLM-L6-v2

# Metrics Configuration
RESPONSE_TIME_THRESHOLD = 5.0  # seconds
SIMILARITY_THRESHOLD = 0.7
