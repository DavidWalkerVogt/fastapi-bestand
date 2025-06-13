
import requests
import pandas as pd
from fastapi import HTTPException

API_BASE = "http://vpc379:8100"
URL_WBZ = f"{API_BASE}/wbz"
URL_DISPO = f"{API_BASE}/dispo"
URL_STOCK = f"{API_BASE}/stockgrouped"

def fetch_data():
    try:
        wbz_resp = requests.get(URL_WBZ)
        dispo_resp = requests.get(URL_DISPO)
        stock_resp = requests.get(URL_STOCK)
        wbz_resp.raise_for_status()
        dispo_resp.raise_for_status()
        stock_resp.raise_for_status()
        return {
            'S_artikel': wbz_resp.json(),
            'MMT_MRP_Account': dispo_resp.json(),
            'MLA_Onhand': stock_resp.json()
        }
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Fehler beim Abruf der API-Daten: {e}")

def transform_data(api_data):
    # Load API data into DataFrames
    s_artikel = pd.DataFrame(api_data.get('S_artikel', []))
    dispo = pd.DataFrame(api_data.get('MMT_MRP_Account', []))
    stock = pd.DataFrame(api_data.get('MLA_Onhand', []))

    # Return empty if any source is missing
    if s_artikel.empty or dispo.empty or stock.empty:
        return pd.DataFrame()

    # Compute WBZ end date
    s_artikel['WBZ'] = pd.to_numeric(s_artikel['WBZ'], errors='coerce').fillna(0).astype(int)
    today = pd.Timestamp.now().normalize()
    s_artikel['EndeWBZ'] = today + pd.to_timedelta(s_artikel['WBZ'], unit='d')
    s_artikel['Datum (Ende WBZ) Num'] = s_artikel['EndeWBZ'].astype(int) // 10**9

    # Filter dispositions within WBZ
    dispo['Termin'] = pd.to_datetime(dispo['Termin'], dayfirst=True, errors='coerce')
    dispo = dispo.merge(s_artikel[['Teil', 'EndeWBZ']], on='Teil', how='left')
    dispo_wbz = dispo[dispo['Termin'] <= dispo['EndeWBZ']]

    # Remove PPA demand/supply pairs (equal demand & supply for same KommNr)
    paired_k = (
        dispo_wbz.groupby('KommNr')
                 .agg({'Bedarfsmenge':'sum','Deckungsmenge':'sum'})
                 .query('Bedarfsmenge>0 and Deckungsmenge>0 and Bedarfsmenge==Deckungsmenge')
                 .index
    )
    dispo_wbz = dispo_wbz[~dispo_wbz['KommNr'].isin(paired_k)]

    # Calculate cumulative demand within WBZ
    bedarf = (
        dispo_wbz[dispo_wbz['Bedarfsmenge'] > 0]
                 .groupby('Teil')['Bedarfsmenge']
                 .sum()
                 .reset_index()
                 .rename(columns={'Bedarfsmenge':'kum Bedarfsmenge'})
    )

    # Calculate cumulative supply within WBZ (ZV/ZL via SubRefObj or PPA via KommNr starting with V-)
    is_zvzl = dispo_wbz['SubRefObj'].str.startswith(('ZV-','ZL-'), na=False)
    is_ppa = dispo_wbz['KommNr'].str.startswith('V-', na=False)
    supply_mask = (dispo_wbz['Deckungsmenge'] > 0) & (is_zvzl | is_ppa)
    deckung = (
        dispo_wbz[supply_mask]
                 .groupby('Teil')['Deckungsmenge']
                 .sum()
                 .reset_index()
                 .rename(columns={'Deckungsmenge':'kum Deckungsmenge'})
    )

    # Prepare on-hand stock
    stock = stock.rename(columns={'Anzahl':'Bestand (Heute)'})

    # Merge and compute today free available
    df = (
        stock
        .merge(bedarf, on='Teil', how='left')
        .merge(deckung, on='Teil', how='left')
        .merge(s_artikel[['Teil','Datum (Ende WBZ) Num']], on='Teil', how='left')
    )
    df[['kum Bedarfsmenge','kum Deckungsmenge']] = df[['kum Bedarfsmenge','kum Deckungsmenge']].fillna(0)
    df['Heute frei verfügbar'] = (
        df['Bestand (Heute)']
        - df['kum Bedarfsmenge']
        + df['kum Deckungsmenge']
    )

    return df[[
        'Teil',
        'Bestand (Heute)',
        'kum Bedarfsmenge',
        'kum Deckungsmenge',
        'Heute frei verfügbar',
        'Datum (Ende WBZ) Num'
    ]]
