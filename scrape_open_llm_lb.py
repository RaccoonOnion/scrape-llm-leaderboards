import os
import requests
import json
import pandas as pd
import re
import logging
from typing import List, Dict, Any, Optional

# --- Configuration ---
CSV_DIR = "csv"
HG_V2_URL = "https://open-llm-leaderboard-open-llm-leaderboard.hf.space/api/leaderboard/formatted"
OUTPUT_CSV = os.path.join(CSV_DIR, "huggingface_v2.csv")

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---

def safe_float_convert(value: Any) -> Optional[float]:
    """Safely convert a value to float, handling potential errors."""
    try: return float(value)
    except (ValueError, TypeError): return None

def safe_int_convert(value: Any) -> Optional[int]:
    """Safely convert a value to int, handling potential errors."""
    try:
        if isinstance(value, float): return int(value)
        if isinstance(value, str):
                value = value.replace(',', '')
                float_val = safe_float_convert(value)
                if float_val is not None:
                    if float_val == int(float_val): return int(float_val)
                    else: return int(float_val)
        return int(value)
    except (ValueError, TypeError): return None

def safe_bool_convert(value: Any) -> Optional[bool]:
    """Safely convert a value to bool, handling potential errors."""
    if isinstance(value, bool): return value
    if isinstance(value, str):
        val_lower = value.lower()
        if val_lower in ['true', '1', 'yes', 't']: return True
        if val_lower in ['false', '0', 'no', 'f']: return False
    if isinstance(value, (int, float)): return bool(value)
    return None

def set_column_types(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt to convert columns to numeric or boolean types."""
    for col in df.columns:
        if col in ['model', 'model_name', 'model_url', 'key'] or df[col].dtype == 'object' and any(isinstance(x, (dict, list)) for x in df[col].dropna()):
            continue # Skip specific text columns and cols with complex objects
        original_series = df[col].copy()
        converted = False
        try: # Boolean
            bool_series = original_series.astype(str).str.lower().map({'true': True, 'false': False, '1': True, '0': False, 'yes': True, 'no': False, 't': True, 'f': False})
            if bool_series.notna().mean() > 0.8 and original_series[bool_series.isna()].nunique() < 5:
                df[col] = bool_series.astype('boolean')
                converted = True
        except Exception: pass
        if converted: continue

        try: # Numeric
            numeric_series = pd.to_numeric(original_series, errors='coerce')
            if numeric_series.notna().sum() > 0.8 * original_series.notna().sum():
                if (numeric_series.dropna() == numeric_series.dropna().round()).all():
                    try: df[col] = numeric_series.astype('Int64'); converted = True
                    except Exception: df[col] = numeric_series.astype(float); converted = True
                else:
                    df[col] = numeric_series.astype(float); converted = True
        except (ValueError, TypeError): pass
    return df

def normalize_hg_headers(headers: List[str]) -> List[str]:
    """Normalizes headers for Hugging Face leaderboard data."""
    normalized = []
    for h in headers:
        h_norm = h.replace('.', '_') # Replace dots from json_normalize
        h_norm = re.sub(r'\s+', '_', h_norm) # Replace spaces
        h_norm = re.sub(r'[^\w_]', '', h_norm) # Remove non-alphanumeric
        normalized.append(h_norm.lower())
    return normalized

def fetch_hg_v2_leaderboard(api_url: str) -> Optional[pd.DataFrame]:
    """Fetches and processes Hugging Face v2 leaderboard data from API."""
    logging.info(f"Fetching Hugging Face v2 leaderboard from API: {api_url}")
    try:
        response = requests.get(api_url, timeout=60)
        response.raise_for_status()
        data = response.json()
        if not data:
                logging.warning("Hugging Face v2 API returned empty data.")
                return None

        df = pd.json_normalize(data) # Flatten nested JSON
        logging.info(f"Normalized Hugging Face v2 data. Initial shape: {df.shape}")

        df.columns = normalize_hg_headers(df.columns.tolist()) # Normalize column names

        # Identify model ID column and create model_url
        model_id_col = None
        potential_id_cols = ['model_info_name', 'model_name', 'model', 'name']
        for col in potential_id_cols:
            if col in df.columns:
                model_id_col = col
                logging.info(f"Using column '{model_id_col}' for model ID.")
                break

        if model_id_col:
            df['model_url'] = df[model_id_col].apply(lambda x: f"https://huggingface.co/{x}" if isinstance(x, str) and '/' in x else None)
            if model_id_col != 'model_name':
                    df.rename(columns={model_id_col: 'model_name'}, inplace=True)
                    if 'model_name' not in df.columns and model_id_col in df.columns:
                        df['model_name'] = df[model_id_col] # Ensure column exists
        else:
                logging.warning("Could not reliably identify model ID column.")
                if 'model_name' not in df.columns: df['model_name'] = None
                if 'model_url' not in df.columns: df['model_url'] = None

        df = set_column_types(df) # Apply type conversions
        logging.info(f"Successfully processed Hugging Face v2 leaderboard. Final shape: {df.shape}")
        return df

    except requests.exceptions.RequestException as e:
        logging.error(f"HTTP request failed for Hugging Face v2: {e}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON from Hugging Face v2: {e}")
        return None
    except Exception as e:
        logging.error(f"An error occurred processing Hugging Face v2 data: {e}", exc_info=True)
        return None

# --- Main Execution ---
if __name__ == "__main__":
    logging.info("--- Starting Hugging Face Scraper ---")
    os.makedirs(CSV_DIR, exist_ok=True) # Ensure dir exists

    df_hg = fetch_hg_v2_leaderboard(HG_V2_URL)

    if df_hg is not None:
        try:
            df_hg.to_csv(OUTPUT_CSV, index=False)
            logging.info(f"Successfully saved Hugging Face data to: {OUTPUT_CSV}")
        except Exception as e:
            logging.error(f"Failed to save Hugging Face DataFrame to CSV {OUTPUT_CSV}: {e}")
    else:
        logging.warning("Fetching Hugging Face function returned None. Cannot save CSV.")

    logging.info("--- Hugging Face Scraper Finished ---")
