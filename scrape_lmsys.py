import os
import requests
import json
import pandas as pd
from bs4 import BeautifulSoup
import re
import logging
from typing import List, Dict, Any, Optional, Tuple

# --- Configuration ---
CSV_DIR = "csv"
LMSYS_URL = "https://lmarena-ai-chatbot-arena-leaderboard.hf.space/"
OUTPUT_CSV = os.path.join(CSV_DIR, "lmsys.csv")

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---

def safe_float_convert(value: Any) -> Optional[float]:
    """Safely convert a value to float, handling potential errors."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def safe_int_convert(value: Any) -> Optional[int]:
    """Safely convert a value to int, handling potential errors."""
    try:
        if isinstance(value, float):
                return int(value)
        if isinstance(value, str):
                value = value.replace(',', '')
                float_val = safe_float_convert(value)
                if float_val is not None:
                    if float_val == int(float_val):
                        return int(float_val)
                    else:
                        return int(float_val)
        return int(value)
    except (ValueError, TypeError):
        return None

def safe_bool_convert(value: Any) -> Optional[bool]:
    """Safely convert a value to bool, handling potential errors."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        val_lower = value.lower()
        if val_lower in ['true', '1', 'yes', 't']: return True
        if val_lower in ['false', '0', 'no', 'f']: return False
    if isinstance(value, (int, float)): return bool(value)
    return None

def extract_model_name_and_url(model_html: str) -> Tuple[Optional[str], Optional[str]]:
    """Parses HTML string to extract model name and URL."""
    if not isinstance(model_html, str) or not model_html.strip(): return None, None
    try:
        soup = BeautifulSoup(model_html, 'lxml')
        link = soup.find('a')
        if link:
            model_name = link.get_text(strip=True)
            model_url = link.get('href')
            if model_url and model_url.startswith('/'):
                    model_url = f"https://huggingface.co{model_url}"
            return model_name, model_url
        else:
                model_name = soup.get_text(strip=True)
                if model_name and '/' in model_name and ' ' not in model_name:
                    model_url = f"https://huggingface.co/{model_name}"
                    return model_name, model_url
                return model_name, None
    except Exception as e:
        logging.warning(f"Could not parse model HTML snippet: {model_html}. Error: {e}")
        try: return BeautifulSoup(model_html, 'lxml').get_text(strip=True), None
        except Exception: return str(model_html), None

def add_model_url_columns(df: pd.DataFrame, html_col: str = 'model') -> pd.DataFrame:
    """Adds 'model_name' and 'model_url' columns by parsing an HTML column."""
    if html_col not in df.columns:
        logging.warning(f"Column '{html_col}' not found for extracting model name/URL.")
        df['model_name'] = None
        df['model_url'] = None
        return df
    results = []
    for item in df[html_col]:
        try: results.append(extract_model_name_and_url(item))
        except Exception as e:
            logging.error(f"Error applying extract_model_name_and_url to item: {item}. Error: {e}")
            results.append((None, None))
    models_urls = pd.DataFrame(results, index=df.index, columns=['model_name', 'model_url'])
    df['model_name'] = models_urls['model_name']
    df['model_url'] = models_urls['model_url']
    return df

def set_column_types(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt to convert columns to numeric or boolean types."""
    for col in df.columns:
        if col in ['model', 'model_name', 'model_url', 'key'] or df[col].dtype == 'object' and any(isinstance(x, (dict, list)) for x in df[col].dropna()):
                continue
        original_series = df[col].copy()
        try: # Boolean
            bool_series = original_series.astype(str).str.lower().map({'true': True, 'false': False, '1': True, '0': False, 'yes': True, 'no': False, 't': True, 'f': False})
            if bool_series.notna().mean() > 0.8 and original_series[bool_series.isna()].nunique() < 5:
                    df[col] = bool_series.astype('boolean')
                    continue
        except Exception: pass
        try: # Numeric
            numeric_series = pd.to_numeric(original_series, errors='coerce')
            if numeric_series.notna().sum() > 0.8 * original_series.notna().sum():
                    if (numeric_series.dropna() == numeric_series.dropna().round()).all():
                        try:
                            df[col] = numeric_series.astype('Int64')
                            converted_numeric = True
                        except Exception:
                            df[col] = numeric_series.astype(float)
                            converted_numeric = True
                    else:
                        df[col] = numeric_series.astype(float)
                        converted_numeric = True
                    if converted_numeric: continue
        except (ValueError, TypeError): pass
    return df

def extract_json_from_html(html_content: str) -> Optional[Dict[str, Any]]:
    """Extracts Gradio config JSON embedded in HTML script tags."""
    soup = BeautifulSoup(html_content, 'lxml')
    scripts = soup.find_all('script')
    for script in scripts:
        if script.string and 'window.gradio_config' in script.string:
            json_str = script.string.strip()
            json_str = json_str.replace('window.gradio_config = ', '', 1)
            if json_str.endswith(';'): json_str = json_str[:-1]
            try: return json.loads(json_str)
            except json.JSONDecodeError as e:
                logging.warning(f"Initial JSON decode failed: {e}. Trying regex extraction.")
                match = re.search(r'window.gradio_config\s*=\s*({(?:[^{}]|{(?:[^{}]|{[^{}]*})*})*});?\s*$', script.string, re.DOTALL | re.MULTILINE)
                if match:
                    json_str_match = match.group(1)
                    try: return json.loads(json_str_match)
                    except json.JSONDecodeError as e_inner:
                            logging.error(f"Retry JSON decode failed: {e_inner}")
                            return None
                else:
                    logging.error("Could not find JSON structure with regex.")
                    return None
    logging.warning("Could not find Gradio config JSON in HTML.")
    return None

def normalize_lmsys_headers(headers: List[str]) -> List[str]:
    """Normalizes headers for LMSYS leaderboard data."""
    normalized = []
    for h in headers:
        h_norm = re.sub(r'\W+', ' ', h).strip()
        if "Rank UB" in h_norm: h_norm = "Rank"
        if "Rank" in h_norm and "StyleCtrl" not in h_norm: h_norm = "Rank"
        if "Arena Elo" in h_norm: h_norm = "Arena_Score"
        if "Arena Score" in h_norm: h_norm = "Arena_Score"
        if "95 CI" in h_norm: h_norm = "95_Pct_CI"
        h_norm = re.sub(r'\s+', '_', h_norm).lower()
        normalized.append(h_norm)
    return normalized

def fetch_lmsys_leaderboard(space_url: str) -> Optional[pd.DataFrame]:
    """Fetches and processes LMSYS leaderboard data from HF Space HTML."""
    logging.info(f"Fetching LMSYS leaderboard from HF Space: {space_url}")
    try:
        response = requests.get(space_url, timeout=60)
        response.raise_for_status()
        html_content = response.text
        config = extract_json_from_html(html_content)
        if not config: return None

        leaderboard_data = None
        leaderboard_headers = None
        if 'components' in config:
                for idx, comp in enumerate(config.get('components', [])):
                    props = comp.get('props', {})
                    value = props.get('value')
                    if isinstance(value, dict) and 'headers' in value and 'data' in value:
                        headers_check = value.get('headers', [])
                        if any(h in headers_check for h in ["Elo", "Arena Score", "Rank", "Votes", "Model"]):
                                leaderboard_data = value.get('data')
                                leaderboard_headers = value.get('headers')
                                logging.info(f"Found LMSYS data via direct value inspection at component index {idx}.")
                                break
                    elif comp.get('type') == 'dataframe' and 'headers' in props and 'value' in props:
                        headers_check = props.get('headers', [])
                        if any(h in headers_check for h in ["Elo", "Arena Score", "Rank", "Votes", "Model"]):
                                leaderboard_data = props.get('value')
                                leaderboard_headers = props.get('headers')
                                logging.info(f"Found LMSYS data via dataframe component props at index {idx}.")
                                break

        if leaderboard_data is None or leaderboard_headers is None:
                logging.error("Could not find LMSYS leaderboard data in the extracted JSON config.")
                return None

        normalized_headers = normalize_lmsys_headers(leaderboard_headers)
        df = pd.DataFrame(leaderboard_data, columns=normalized_headers)
        df = add_model_url_columns(df, html_col='model')

        if 'model_name' in df.columns:
                df['model'] = df['model_name']
                logging.info("Replaced 'model' column with extracted text name.")
        else:
                logging.warning("Could not replace 'model' column as 'model_name' was not found.")

        df = set_column_types(df)
        logging.info(f"Successfully processed LMSYS leaderboard. Shape: {df.shape}")
        return df

    except requests.exceptions.RequestException as e:
        logging.error(f"HTTP request failed for LMSYS: {e}")
        return None
    except Exception as e:
        logging.error(f"An error occurred processing LMSYS data: {e}", exc_info=True)
        return None

# --- Main Execution ---
if __name__ == "__main__":
    logging.info("--- Starting LMSYS Scraper ---")
    os.makedirs(CSV_DIR, exist_ok=True) # Ensure dir exists

    df_lmsys = fetch_lmsys_leaderboard(LMSYS_URL)

    if df_lmsys is not None:
        try:
            df_lmsys.to_csv(OUTPUT_CSV, index=False)
            logging.info(f"Successfully saved LMSYS data to: {OUTPUT_CSV}")
        except Exception as e:
            logging.error(f"Failed to save LMSYS DataFrame to CSV {OUTPUT_CSV}: {e}")
    else:
        logging.warning("Fetching LMSYS function returned None. Cannot save CSV.")

    logging.info("--- LMSYS Scraper Finished ---")
