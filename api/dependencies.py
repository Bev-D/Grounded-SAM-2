# api/dependencies.py
from fastapi import Depends
from fastapi import FastAPI

def get_processor(app: FastAPI = Depends()):
    return app.state.processor
