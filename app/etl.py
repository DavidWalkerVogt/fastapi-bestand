import requests
import pandas as pd
from datetime import datetime
from fastapi import HTTPException

API_BASE = "http://vpc379:8100"
URL_WBZ = f"{API_BASE}/wbz"
URL_DISPO = f"{API_BASE}/dispo"
URL_STOCK = f"{API_BASE}/stockgrouped"

def fetch_data():
    try:
        wbz = requests.get(URL_WBZ)
        dispo = requests.get(URL_DISPO)
        stock = requests.get(URL_STOCK)

        wbz.raise_for_status()
        dispo.raise_for_status()
        stock.raise_for_status()

        return {
            "S_artikel": wbz.json(),
            "MMT_MRP_Account": dispo.json(),
            "MLA_Onhand": stock.json()
        }
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Fehler beim Abruf der API-Daten: {e}")

def transform_data(api_data):
    s_artikel = pd.DataFrame(api_data["S_artikel"])
    dispo = pd.DataFrame(api_data["MMT_MRP_Account"])
    stock = pd.DataFrame(api_data["MLA_Onhand"])

    if s_artikel.empty or dispo.empty or stock.empty:
        return pd.DataFrame()

    dispo["ZV_or_ZL"] = dispo["SubRefObj"].str[:2].isin(["ZV", "ZL"])
    dispo["Bestands relevant"] = dispo["ZV_or_ZL"].astype(int)

    dispo["Termin"] = pd.to_datetime(dispo["Termin"], errors="coerce")
    dispo.rename(columns={"Termin": "Datum Numeric"}, inplace=True)
    s_artikel["Datum (Ende WBZ) Num"] = pd.to_datetime(s_artikel["WBZ"], errors="coerce")

    bedarf = dispo[dispo["Bestands relevant"] == 1].groupby("Teil")["Bedarfsmenge"].sum().reset_index()
    bedarf = bedarf.rename(columns={"Bedarfsmenge": "kum Bedarfsmenge"})

    deckung = dispo[dispo["ZV_or_ZL"] == True].groupby("Teil")["Deckungsmenge"].sum().reset_index()
    deckung = deckung.rename(columns={"Deckungsmenge": "kum Deckungsmenge"})

    stock = stock.rename(columns={"Anzahl": "Bestand (Heute)"})

    df = pd.merge(stock, bedarf, on="Teil", how="outer")
    df = pd.merge(df, deckung, on="Teil", how="outer")
    df = pd.merge(df, s_artikel, on="Teil", how="outer")

    df.fillna(0, inplace=True)

    df["Heute frei verfügbar"] = (
        df["Bestand (Heute)"] - df["kum Bedarfsmenge"] + df["kum Deckungsmenge"]
    )

    return df[[
        "Teil",
        "Bestand (Heute)",
        "kum Bedarfsmenge",
        "kum Deckungsmenge",
        "Heute frei verfügbar",
        "Datum (Ende WBZ) Num"
    ]]
