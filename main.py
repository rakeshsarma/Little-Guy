from fastapi import FastAPI
from langcorn import create_service

app = create_service(
    "ConvBot:little_guy_with_memory"
)
