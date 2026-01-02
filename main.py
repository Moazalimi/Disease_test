import sys
import os

# Run app.py
if __name__ == "__main__":
    # For testing: uvicorn main:app --reload
    # Or directly: python -m uvicorn app:app --reload
    from app import app
