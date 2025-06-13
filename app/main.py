from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from app.etl import fetch_data, transform_data

app = FastAPI(title="Bestands-API")

class ArticleRequest(BaseModel):
    article: List[str]

@app.post("/calculate", summary="Berechne nur bestimmte Artikel")
def calculate_bestand(request: ArticleRequest):
    api_data = fetch_data()
    df       = transform_data(api_data)
    filtered = df[df['Teil'].isin(request.article)]
    return filtered.to_dict(orient="records")

@app.get("/calculate_all", summary="Berechne alle Artikel")
def calculate_all():
    api_data = fetch_data()
    df       = transform_data(api_data)
    return df.to_dict(orient="records")
