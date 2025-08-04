from typing import List
from fastapi import FastAPI
from pydantic import BaseModel

from .etl import fetch_data, fetch_data_with_debug

app = FastAPI(
    title="Verfügbarkeits-API",
    description="Berechnet den heute frei verfügbaren Bestand gemäss Pflichtenheft (WBZ, AUF/EBE/PPA).",
    version="1.4.0",
)

class ArticleRequest(BaseModel):
    article: List[str]

@app.post("/calculate", summary="Berechne bestimmte Artikel")
async def calculate_bestand(request: ArticleRequest):
    df = fetch_data()
    return df[df["Teil"].isin(request.article)].to_dict(orient="records")

@app.get("/calculate_all", summary="Berechne alle Artikel")
async def calculate_all():
    return fetch_data().to_dict(orient="records")

@app.get("/debug/{teil}", summary="Debug-Ausgabe für einen Artikel")
async def debug_teil(teil: str):
    df_res, dbg = fetch_data_with_debug(teil)
    return {"result": df_res.to_dict(orient="records"), "debug": dbg}

@app.get("/health", summary="Health-Check")
def health_check():
    return {"status": "ok"}