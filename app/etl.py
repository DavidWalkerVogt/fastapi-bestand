from __future__ import annotations
import os, re
from pathlib import Path
from datetime import date
from typing import Any, Dict, List, Tuple
import csv

import numpy as np
import pandas as pd
import requests
from fastapi import HTTPException

# API Endpoints
API_BASE  = "http://vpc379:8100"
URL_WBZ   = f"{API_BASE}/wbz"
URL_DISPO = f"{API_BASE}/dispo"
URL_STOCK = f"{API_BASE}/stockgrouped"

# Werktagsdefinition für WBZ
WEEKMASK = "Mon Tue Wed Thu Fri"

# ---------------------- Helper ----------------------
def _clean_part_series(sr: pd.Series) -> pd.Series:
    return (
        sr.astype(str)
          .str.replace(r"[\r\n\t]", "", regex=True)
          .str.replace("\ufeff", "", regex=False)
          .str.replace("\xa0", "", regex=False)
          .str.replace('"', "", regex=False)
          .str.strip()
          .str.lower()
    )

def _find_teil_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if "teil" in col.lower():
            return col
    return None

def _normalize_parts(df: pd.DataFrame) -> None:
    col = _find_teil_column(df)
    if not col:
        return
    if col != "Teil":
        df.rename(columns={col: "Teil"}, inplace=True)
    df["Teil_norm"] = _clean_part_series(df["Teil"])

def ensure_str_col(df: pd.DataFrame, col: str) -> None:
    """
    Ensure that column `col` exists as string type in DataFrame df.
    If missing, create with empty strings; else fill NaNs and cast to str.
    """
    if col not in df.columns:
        df[col] = ""
    else:
        df[col] = df[col].fillna("").astype(str)

def _numcol(df: pd.DataFrame, primary: str, fallbacks: List[str]) -> pd.Series:
    for c in [primary] + fallbacks:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce").fillna(0)
    return pd.Series(0, index=df.index, dtype="float64")

def busday_add(start: np.datetime64, days: int) -> pd.Timestamp:
    end = np.busday_offset(start, days, roll="forward", weekmask=WEEKMASK)
    return pd.Timestamp(end)

def calc_wbz_end(df: pd.DataFrame, wbz_days: int, today: np.datetime64) -> pd.Timestamp:
    if "Beleginfo" in df.columns:
        mask = df["Beleginfo"].str.contains("WBZ DisB", case=False, na=False)
        wbz_rows = df[mask]
        if not wbz_rows.empty:
            pref = wbz_rows[wbz_rows["Beleginfo"].str.contains("DisB 0", case=False, na=False)]
            row = pref.iloc[0] if not pref.empty else wbz_rows.iloc[0]
            return pd.to_datetime(row["Termin"])
    return busday_add(today, wbz_days)

def _fix_stock(df: pd.DataFrame) -> pd.DataFrame:
    # Reparatur stockgrouped.csv falls Werte in einer Spalte gespeichert
    if "Anzahl" in df.columns and df["Anzahl"].notna().any():
        return df
    if df.shape[1] == 1:
        raw = df.iloc[:,0].astype(str).str.replace("\ufeff","",regex=False).str.strip()
        pat = re.compile(r'^"?\s*([^,;]+)\s*[,;]\s*"?([\d\.]+)"?.*$')
        extr = raw.str.extract(pat)
        extr.columns = ["Teil","Anzahl"]
        df = extr
    df["Anzahl"] = pd.to_numeric(df.get("Anzahl",0), errors="coerce").fillna(0)
    return df

# ---------------------- Core Computation ----------------------
def compute_for_part(
    teil_norm: str,
    dispo_grp: pd.DataFrame,
    wbz_days: int,
    stock_qty: float,
    today: np.datetime64
) -> Dict[str, Any]:
    # WBZ-Ende bestimmen
    wbz_end = calc_wbz_end(dispo_grp, wbz_days, today)
    # Dispo-Zeilen innerhalb WBZ (Termine vor wbz_end)
    in_wbz = dispo_grp.dropna(subset=['Termin'])
    in_wbz = in_wbz[in_wbz['Termin'] <= wbz_end]
    # Deckungsmengen (Eingänge) summieren
    total_cover = in_wbz['Deckungsmenge'].sum() if 'Deckungsmenge' in in_wbz.columns else 0.0
    # Bedarfsmenge (Verbräuche) summieren
    total_demand = in_wbz['Bedarfsmenge'].sum() if 'Bedarfsmenge' in in_wbz.columns else 0.0
    # Freier Bestand: Bestand + Eingänge - Verbräuche
    heute_frei = float(stock_qty) + float(total_cover) - float(total_demand)
    # Ausgabe Teilbezeichnung
    teil_out = dispo_grp['Teil'].iloc[0] if not dispo_grp.empty and 'Teil' in dispo_grp.columns else teil_norm
    # Ergebnis zurückgeben
    return {
        'Teil': teil_out,
        'Bestand (Heute)': float(stock_qty),
        'Zugänge (WBZ)': float(total_cover),
        'Verbräuche (WBZ)': float(total_demand),
        'Heute frei verfügbar': heute_frei,
        'WBZ_Ende': wbz_end,
    }

# ---------------------- Fetch & Prepare ---------------------- & Prepare ----------------------
def _fetch_sources() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    use_local = os.getenv("USE_LOCAL_DATA","0")=="1"
    if use_local:
        base = Path(os.getenv("LOCAL_DATA_DIR","./data"))
        # WBZ lokal
        df_w = pd.read_csv(base/"wbz.csv", encoding="utf-8-sig")
        # Dispo.csv manuell parsen
        dispo_path = base/"dispo.csv"
        lines = dispo_path.read_text(encoding="utf-8-sig").splitlines()
        if not lines:
            df_d = pd.DataFrame()
        else:
            raw_header = lines[0]
            if raw_header.startswith('"') and raw_header.endswith('"'):
                core_h = raw_header[1:-1].replace('""','"')
            else:
                core_h = raw_header
            cols = next(csv.reader([core_h], quotechar='"'))
            records: List[Dict[str, Any]] = []
            for line in lines[1:]:
                if not line:
                    continue
                if line.startswith('"') and line.endswith('"'):
                    core = line[1:-1].replace('""','"')
                else:
                    core = line
                try:
                    vals = next(csv.reader([core], quotechar='"'))
                except Exception:
                    continue
                if len(vals)==len(cols):
                    records.append(dict(zip(cols,vals)))
            df_d = pd.DataFrame(records)
        # Stock lokal
        df_s = pd.read_csv(base/"stockgrouped.csv", encoding="utf-8-sig")
    else:
        try:
            r1=requests.get(URL_WBZ); r1.raise_for_status()
            r2=requests.get(URL_DISPO); r2.raise_for_status()
            r3=requests.get(URL_STOCK); r3.raise_for_status()
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))
        df_w = pd.DataFrame(r1.json())
        df_d = pd.DataFrame(r2.json())
        df_s = pd.DataFrame(r3.json())
    df_s = _fix_stock(df_s)
    for df in (df_w, df_d, df_s):
        df.columns = df.columns.str.strip().str.replace('"','',regex=False)
        _normalize_parts(df)
    return df_w, df_d, df_s

def _prepare_frames(
    df_w: pd.DataFrame,
    df_d: pd.DataFrame,
    df_s: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # WBZ-Tage
    df_w["WBZ"] = _numcol(df_w, "WBZ", ["wbz", "wbz_tage"]).astype(int)
    # Dispo: Deckung und Bedarf numeric
    df_d["Deckungsmenge"] = _numcol(df_d, "Deckungsmenge", ["Deckungmenge", "Deckung_Menge"])
    df_d["Bedarfsmenge"]  = _numcol(df_d, "Bedarfsmenge", ["Bedarfmenge", "Bedarf_Menge"])
    df_d["Termin"]        = pd.to_datetime(df_d.get("Termin"), errors="coerce")
    # Stock
    df_s["Anzahl"] = _numcol(df_s, "Anzahl", ["Bestand", "Bestand (Heute)"])
    # String columns in dispo
    for col in ["Herkunftsbelegart", "HBA", "Beleginfo", "KommNr"]:
        ensure_str_col(df_d, col)
    return df_w, df_d, df_s

# ---------------------- Public API ----------------------
def fetch_data() -> pd.DataFrame:
    df_w, df_d, df_s = _fetch_sources()
    df_w, df_d, df_s = _prepare_frames(df_w, df_d, df_s)
    today = np.datetime64(date.today(),'D')
    key = 'Teil_norm'
    wbz_map   = df_w.set_index(key)['WBZ'].to_dict() if key in df_w else {}
    stock_map = df_s.set_index(key)['Anzahl'].to_dict() if key in df_s else {}
    results: List[Dict[str,Any]] = []
    parts = set(df_d[key]).union(stock_map.keys(), wbz_map.keys())
    for part in parts:
        grp = df_d[df_d[key]==part]
        stock = float(stock_map.get(part,0))
        days  = int(wbz_map.get(part,0))
        results.append(compute_for_part(part,grp,days,stock,today))
    return pd.DataFrame(results)

def fetch_data_with_debug(teil: str) -> Tuple[pd.DataFrame,Dict[str,Any]]:
    df_w, df_d, df_s = _fetch_sources()
    df_w, df_d, df_s = _prepare_frames(df_w, df_d, df_s)
    today   = np.datetime64(date.today(),'D')
    clean   = _clean_part_series(pd.Series([teil])).iloc[0]
    key     = 'Teil_norm'
    stock   = float(df_s.set_index(key)['Anzahl'].get(clean,0))
    days    = int(df_w.set_index(key)['WBZ'].get(clean,0))
    grp     = df_d[df_d[key]==clean]
    wbz_end = calc_wbz_end(grp,days,today)
    inner   = grp[(grp['Termin'].notna())&(grp['Termin']<=wbz_end)]
    covers  = inner[inner.get('Deckungsmenge',0)>0]
    total   = float(covers['Deckungsmenge'].sum())
    out_df  = pd.DataFrame([compute_for_part(clean,grp,days,stock,today)])
    debug   = {
        'teil_clean': clean,
        'wbz_days': days,
        'wbz_end': str(wbz_end),
        'bestand_heute': stock,
        'covers_rows': covers.to_dict('records'),
        'totals': {'cover': total},
        'stock_rows_raw': df_s[df_s[key]==clean].to_dict('records'),
        'dispo_rows_raw': df_d[df_d[key]==clean].to_dict('records'),
        'df_w_columns': df_w.columns.tolist(),
        'df_d_columns': df_d.columns.tolist(),
        'df_s_columns': df_s.columns.tolist(),
    }
    return out_df, debug
