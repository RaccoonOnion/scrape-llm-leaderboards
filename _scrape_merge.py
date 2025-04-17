import os
import requests
import json
import pandas as pd
from bs4 import BeautifulSoup
import re
import logging
from typing import List, Dict, Any, Optional, Tuple

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create csv directory if it doesn't exist
CSV_DIR = "csv"
os.makedirs(CSV_DIR, exist_ok=True)
logging.info(f"Ensured '{CSV_DIR}' directory exists.")

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
        # Handle cases like '7.242' if they should be truncated to int
        if isinstance(value, float):
             return int(value)
        # Handle string representations of floats before converting to int
        if isinstance(value, str):
             # Remove commas for thousands separators if present
             value = value.replace(',', '')
             float_val = safe_float_convert(value)
             if float_val is not None:
                 # Check if it's effectively an integer
                 if float_val == int(float_val):
                     return int(float_val)
                 else:
                      # If it has a decimal part, maybe return None or raise error?
                      # For now, let's allow truncation if called directly
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
        if val_lower in ['true', '1', 'yes', 't']:
            return True
        if val_lower in ['false', '0', 'no', 'f']:
            return False
    # Consider numeric representations? e.g., 1 = True, 0 = False
    if isinstance(value, (int, float)):
         return bool(value) # Standard Python bool conversion for numbers
    return None # Or return False by default? Depends on desired behavior

def extract_model_name_and_url(model_html: str) -> Tuple[Optional[str], Optional[str]]:
    """Parses HTML string to extract model name and URL."""
    if not isinstance(model_html, str) or not model_html.strip():
        return None, None
    try:
        soup = BeautifulSoup(model_html, 'lxml')
        link = soup.find('a')
        if link:
            model_name = link.get_text(strip=True)
            model_url = link.get('href')
            # Prepend base URL if href is relative (common in HF spaces)
            if model_url and model_url.startswith('/'):
                 # Attempt to guess base URL - this might need adjustment
                 # For HF spaces, it's usually huggingface.co
                 model_url = f"https://huggingface.co{model_url}"
            return model_name, model_url
        else:
            # Handle cases where the input is just text (no link)
             model_name = soup.get_text(strip=True)
             # Sometimes the text itself might be a simple model name like 'GPT-4'
             # If it looks like a HF model ID, construct a potential URL
             if model_name and '/' in model_name and ' ' not in model_name:
                 model_url = f"https://huggingface.co/{model_name}"
                 return model_name, model_url
             return model_name, None

    except Exception as e:
        logging.warning(f"Could not parse model HTML snippet: {model_html}. Error: {e}")
        # Attempt to return the raw text if parsing fails
        try:
            raw_text = BeautifulSoup(model_html, 'lxml').get_text(strip=True)
            return raw_text, None
        except Exception:
             return str(model_html), None # Fallback to original string


def add_model_url_columns(df: pd.DataFrame, html_col: str = 'model') -> pd.DataFrame:
    """Adds 'model_name' and 'model_url' columns by parsing an HTML column."""
    if html_col not in df.columns:
        logging.warning(f"Column '{html_col}' not found for extracting model name/URL.")
        df['model_name'] = None
        df['model_url'] = None
        return df

    # Apply extraction; handle potential errors during apply
    results = []
    for item in df[html_col]:
        try:
            results.append(extract_model_name_and_url(item))
        except Exception as e:
            logging.error(f"Error applying extract_model_name_and_url to item: {item}. Error: {e}")
            results.append((None, None)) # Append Nones on error

    models_urls = pd.DataFrame(results, index=df.index, columns=['model_name', 'model_url'])

    df['model_name'] = models_urls['model_name']
    df['model_url'] = models_urls['model_url']

    # Overwrite original HTML column with extracted name if requested
    # df[html_col] = df['model_name'] # Moved this logic to fetch_lmsys_leaderboard

    return df


def set_column_types(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt to convert columns to numeric or boolean types."""
    for col in df.columns:
        # Skip columns we know shouldn't be converted or are already processed
        if col in ['model', 'model_name', 'model_url', 'key'] or df[col].dtype == 'object' and any(isinstance(x, (dict, list)) for x in df[col].dropna()):
             # Skip key text columns and columns that still contain complex objects
             continue

        original_series = df[col].copy() # Keep original for comparison/fallback

        # 1. Try Boolean Conversion First (more specific patterns)
        try:
            # Use nullable Boolean type
            bool_series = original_series.astype(str).str.lower().map({'true': True, 'false': False, '1': True, '0': False, 'yes': True, 'no': False, 't': True, 'f': False})
            # Only apply if a significant portion could be converted AND few unique non-bool values remain
            if bool_series.notna().mean() > 0.8 and original_series[bool_series.isna()].nunique() < 5:
                 df[col] = bool_series.astype('boolean')
                 # logging.debug(f"Column '{col}' converted to Boolean.")
                 continue # Skip to next column if successfully converted to bool
        except Exception:
            pass # Ignore errors and try next conversion

        # 2. Try Numeric Conversion (Int then Float)
        converted_numeric = False
        try:
            # Attempt conversion to numeric, coercing errors to NaN temporarily
            numeric_series = pd.to_numeric(original_series, errors='coerce')

            # Check if conversion was successful for a majority of non-null values
            if numeric_series.notna().sum() > 0.8 * original_series.notna().sum():
                 # Check if integer conversion is appropriate (no data loss)
                 # Use Int64 (nullable integer)
                 if (numeric_series.dropna() == numeric_series.dropna().round()).all():
                     try:
                          df[col] = numeric_series.astype('Int64')
                          # logging.debug(f"Column '{col}' converted to Int64.")
                          converted_numeric = True
                     except Exception: # Fallback to float if Int64 fails for some reason
                          df[col] = numeric_series.astype(float)
                          # logging.debug(f"Column '{col}' converted to Float (fallback from Int64).")
                          converted_numeric = True

                 else:
                     # Keep as float if decimals are present
                     df[col] = numeric_series.astype(float)
                     # logging.debug(f"Column '{col}' converted to Float.")
                     converted_numeric = True

        except (ValueError, TypeError):
            pass # Ignore errors if numeric conversion fails

        # 3. Fallback (Keep as Object/String) - No explicit action needed

    return df


# --- Hugging Face Leaderboard Functions ---

def normalize_hg_headers(headers: List[str]) -> List[str]:
    """Normalizes headers for Hugging Face leaderboard data."""
    normalized = []
    for h in headers:
        h_norm = h
        # Handle specific cases from R script if needed (less relevant after json_normalize)
        # if "Average ⬆️" in h: h_norm = "Average"
        # if "Hub ❤️" in h: h_norm = "Hub_Hearts"

        # General normalization for flattened keys
        h_norm = h_norm.replace('.', '_') # Replace dots from json_normalize with underscores
        h_norm = re.sub(r'\s+', '_', h_norm) # Replace spaces with underscores
        h_norm = re.sub(r'[^\w_]', '', h_norm) # Remove non-alphanumeric (allow underscore)
        normalized.append(h_norm.lower())
    return normalized

def fetch_hg_v2_leaderboard(api_url: str) -> Optional[pd.DataFrame]:
    """Fetches and processes Hugging Face v2 leaderboard data from API."""
    logging.info(f"Fetching Hugging Face v2 leaderboard from API: {api_url}")
    try:
        response = requests.get(api_url, timeout=60)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        if not data:
             logging.warning("Hugging Face v2 API returned empty data.")
             return None

        # --- MODIFICATION: Use json_normalize to flatten nested structures ---
        df = pd.json_normalize(data)
        logging.info(f"Normalized Hugging Face v2 data. Initial shape: {df.shape}")
        # logging.debug(f"Columns after normalize: {df.columns.tolist()}") # Log columns for debugging

        # --- MODIFICATION: Normalize flattened column names ---
        df.columns = normalize_hg_headers(df.columns.tolist())
        # logging.debug(f"Columns after normalize and rename: {df.columns.tolist()}") # Log columns for debugging


        # --- MODIFICATION: Adapt model URL creation and renaming ---
        # Guess the column containing the model ID after normalization (e.g., 'model_info_name')
        model_id_col = None
        potential_id_cols = ['model_info_name', 'model_name', 'model', 'name'] # Add likely candidates
        for col in potential_id_cols:
            if col in df.columns:
                model_id_col = col
                logging.info(f"Using column '{model_id_col}' for model ID.")
                break

        if model_id_col:
            df['model_url'] = df[model_id_col].apply(lambda x: f"https://huggingface.co/{x}" if isinstance(x, str) and '/' in x else None)
            # Rename the identified model ID column to 'model_name' for consistency if it's not already named that
            if model_id_col != 'model_name':
                 df.rename(columns={model_id_col: 'model_name'}, inplace=True)
                 # Ensure 'model_name' column exists even if rename didn't happen (e.g., was already 'model_name')
                 if 'model_name' not in df.columns and model_id_col in df.columns:
                      df['model_name'] = df[model_id_col]

        else:
             logging.warning("Could not reliably identify model ID column in HG v2 API response after normalization.")
             # Ensure columns exist even if creation failed
             if 'model_name' not in df.columns: df['model_name'] = None
             if 'model_url' not in df.columns: df['model_url'] = None


        # Apply type conversions (should handle flattened columns better now)
        df = set_column_types(df)
        logging.info(f"Successfully processed Hugging Face v2 leaderboard. Final shape: {df.shape}")
        return df

    except requests.exceptions.RequestException as e:
        logging.error(f"HTTP request failed for Hugging Face v2: {e}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON from Hugging Face v2: {e}")
        return None
    except Exception as e:
        logging.error(f"An error occurred processing Hugging Face v2 data: {e}", exc_info=True) # Add traceback
        return None

# --- LMSYS Chatbot Arena Functions ---

def extract_json_from_html(html_content: str) -> Optional[Dict[str, Any]]:
    """Extracts Gradio config JSON embedded in HTML script tags."""
    soup = BeautifulSoup(html_content, 'lxml')
    scripts = soup.find_all('script')
    for script in scripts:
        if script.string and 'window.gradio_config' in script.string:
            json_str = script.string.strip()
            # Clean the string to make it valid JSON
            json_str = json_str.replace('window.gradio_config = ', '', 1)
            # Remove trailing semicolon if exists
            if json_str.endswith(';'):
                json_str = json_str[:-1]
            try:
                # Try standard parsing
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logging.warning(f"Initial JSON decode failed: {e}. Trying regex extraction.")
                # Attempt to find JSON boundaries more carefully using regex
                # This regex tries to match balanced braces/brackets
                match = re.search(r'window.gradio_config\s*=\s*({(?:[^{}]|{(?:[^{}]|{[^{}]*})*})*});?\s*$', script.string, re.DOTALL | re.MULTILINE)
                if match:
                    json_str_match = match.group(1)
                    try:
                        # print(f"Trying matched JSON: {json_str_match[:200]}...") # Debug print
                        return json.loads(json_str_match)
                    except json.JSONDecodeError as e_inner:
                         logging.error(f"Retry JSON decode failed: {e_inner}")
                         return None
                else:
                    logging.error("Could not find JSON structure with regex.")
                    return None
    logging.warning("Could not find Gradio config JSON in HTML.")
    return None

# Note: find_component_by_prop and find_leaderboard_data_index might not be needed
# if the direct value inspection logic in fetch_lmsys_leaderboard works reliably.

def normalize_lmsys_headers(headers: List[str]) -> List[str]:
    """Normalizes headers for LMSYS leaderboard data."""
    normalized = []
    for h in headers:
        h_norm = re.sub(r'\W+', ' ', h).strip() # Replace non-word chars with space, trim
        # Apply specific R script normalizations
        if "Rank UB" in h_norm: h_norm = "Rank" # R code used Rank UB, maybe it's just Rank now?
        if "Rank" in h_norm and "StyleCtrl" not in h_norm: h_norm = "Rank" # Generalize Rank
        if "Arena Elo" in h_norm: h_norm = "Arena_Score" # Match R script output? Or keep elo? Let's use score.
        if "Arena Score" in h_norm: h_norm = "Arena_Score"
        if "95 CI" in h_norm: h_norm = "95_Pct_CI" # Use underscore
        # General replacements
        h_norm = re.sub(r'\s+', '_', h_norm) # Replace spaces with underscores
        h_norm = h_norm.lower()
        normalized.append(h_norm)
    return normalized

def fetch_lmsys_leaderboard(space_url: str) -> Optional[pd.DataFrame]:
    """Fetches and processes LMSYS leaderboard data from HF Space HTML."""
    logging.info(f"Fetching LMSYS leaderboard from HF Space: {space_url}")
    try:
        response = requests.get(space_url, timeout=60)
        response.raise_for_status()
        html_content = response.text

        # Extract the Gradio JSON config
        config = extract_json_from_html(html_content)
        if not config:
            return None

        # Find the leaderboard data within the config
        leaderboard_data = None
        leaderboard_headers = None

        # Try finding the data structure: dict with 'headers' and 'data' keys
        # Iterate through components and their properties
        if 'components' in config:
             for idx, comp in enumerate(config.get('components', [])):
                   props = comp.get('props', {})
                   value = props.get('value')
                   # Check if the value itself is the dict {headers: [], data: [[]]}
                   if isinstance(value, dict) and 'headers' in value and 'data' in value:
                       headers_check = value.get('headers', [])
                       # Use heuristic header names
                       if any(h in headers_check for h in ["Elo", "Arena Score", "Rank", "Votes", "Model"]):
                             leaderboard_data = value.get('data')
                             leaderboard_headers = value.get('headers')
                             logging.info(f"Found LMSYS data via direct value inspection at component index {idx}.")
                             break
                   # Fallback: Check if component type is dataframe and props has headers/value
                   elif comp.get('type') == 'dataframe' and 'headers' in props and 'value' in props:
                        headers_check = props.get('headers', [])
                        if any(h in headers_check for h in ["Elo", "Arena Score", "Rank", "Votes", "Model"]):
                             leaderboard_data = props.get('value') # Value should be list of lists
                             leaderboard_headers = props.get('headers')
                             logging.info(f"Found LMSYS data via dataframe component props at index {idx}.")
                             break

        if leaderboard_data is None or leaderboard_headers is None:
             logging.error("Could not find LMSYS leaderboard data in the extracted JSON config.")
             return None

        # Normalize headers
        normalized_headers = normalize_lmsys_headers(leaderboard_headers)

        # Create DataFrame
        df = pd.DataFrame(leaderboard_data, columns=normalized_headers)

        # Extract model name and URL from the 'model' column (assuming it contains HTML)
        # This adds 'model_name' and 'model_url' columns
        df = add_model_url_columns(df, html_col='model')

        # --- MODIFICATION: Overwrite original 'model' column with clean name ---
        if 'model_name' in df.columns:
             df['model'] = df['model_name']
             logging.info("Replaced 'model' column with extracted text name.")
        else:
             logging.warning("Could not replace 'model' column as 'model_name' was not found.")


        # Apply type conversions
        df = set_column_types(df)

        logging.info(f"Successfully processed LMSYS leaderboard. Shape: {df.shape}")
        return df

    except requests.exceptions.RequestException as e:
        logging.error(f"HTTP request failed for LMSYS: {e}")
        return None
    except Exception as e:
        logging.error(f"An error occurred processing LMSYS data: {e}", exc_info=True) # Add traceback
        return None


# --- Merging Function ---

def merge_leaderboards(df_hg: pd.DataFrame, df_lmsys: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Merges Hugging Face and LMSYS leaderboards on a derived key."""
    if df_hg is None or df_lmsys is None:
        logging.warning("One or both dataframes are None, cannot merge.")
        return None

    # Create merge keys - based on R script logic, adapted for flattened HG data
    # HG: Use 'model_name', take part after '/', lowercase
    if 'model_name' in df_hg.columns:
         df_hg['key'] = df_hg['model_name'].astype(str).str.split('/').str[-1].str.lower().str.strip()
         # Handle cases where model_name might not have '/'
         df_hg['key'] = df_hg.apply(lambda row: row['model_name'].split('/')[-1].lower().strip() if isinstance(row['model_name'], str) and '/' in row['model_name'] else str(row['model_name']).lower().strip(), axis=1)

    else:
         logging.error("Cannot create merge key for Hugging Face data: 'model_name' column missing.")
         return None

    # LMSYS: Use 'model_name' (extracted from HTML), lowercase
    if 'model_name' in df_lmsys.columns:
         # Ensure model_name is string before lowercasing
         df_lmsys['key'] = df_lmsys['model_name'].astype(str).str.lower().str.strip()
    else:
         logging.error("Cannot create merge key for LMSYS data: 'model_name' column missing.")
         return None

    # --- Debugging Keys ---
    # logging.debug(f"HG Keys sample: {df_hg['key'].unique()[:20]}")
    # logging.debug(f"LMSYS Keys sample: {df_lmsys['key'].unique()[:20]}")
    # Find common keys
    common_keys = set(df_hg['key'].dropna()) & set(df_lmsys['key'].dropna())
    logging.info(f"Found {len(common_keys)} common keys for merging.")
    if len(common_keys) < 10: # Log some common keys if count is low
         logging.info(f"Sample common keys: {list(common_keys)[:10]}")
    # --- End Debugging ---


    # Perform the merge
    logging.info("Merging Hugging Face v2 and LMSYS data...")
    # Use inner merge like R's default merge()
    merged_df = pd.merge(df_hg, df_lmsys, on='key', how='inner', suffixes=('_hg', '_lmsys'))

    # Optional: Reorder columns or clean up merged dataframe
    # Bring key columns, model names to the front
    key_cols = ['key', 'model_name_hg', 'model_url_hg', 'model_name_lmsys', 'model_url_lmsys']
    # Filter out any key columns that might not exist if merge failed partially
    key_cols = [col for col in key_cols if col in merged_df.columns]
    other_cols = [col for col in merged_df.columns if col not in key_cols]

    merged_df = merged_df[key_cols + other_cols]


    logging.info(f"Successfully merged leaderboards. Shape: {merged_df.shape}")
    if merged_df.empty:
         logging.warning("Merged dataframe is empty. Check merge keys and source data.")
    return merged_df


# --- Main Execution Logic ---

def dt_if_missing(filename: str, fetch_func, *args, **kwargs) -> Optional[pd.DataFrame]:
    """Loads data from CSV if exists, otherwise fetches using fetch_func."""
    csv_path = os.path.join(CSV_DIR, filename)
    if os.path.exists(csv_path):
        logging.info(f"Loading existing data from: {csv_path}")
        try:
            # Read CSV, try to infer types but be cautious
            return pd.read_csv(csv_path)
        except Exception as e:
            logging.error(f"Failed to load CSV {csv_path}: {e}. Will attempt to refetch.")
            # Optionally delete the corrupt file:
            # try:
            #     os.remove(csv_path)
            # except OSError as rm_err:
            #     logging.error(f"Failed to remove potentially corrupt file {csv_path}: {rm_err}")

    logging.info(f"Data not found locally ({csv_path}). Fetching...")
    df = fetch_func(*args, **kwargs)
    if df is not None:
        try:
            df.to_csv(csv_path, index=False)
            logging.info(f"Successfully fetched and saved data to: {csv_path}")
        except Exception as e:
            logging.error(f"Failed to save DataFrame to CSV {csv_path}: {e}")
    else:
         logging.warning(f"Fetching function returned None for {filename}. Cannot save.")
    return df


if __name__ == "__main__":
    # Define URLs (based on R script and search results)
    # HG v2 API URL from R script
    hg_v2_url = "https://open-llm-leaderboard-open-llm-leaderboard.hf.space/api/leaderboard/formatted"
    # LMSYS HF Space URL from R script
    lmsys_url = "https://lmarena-ai-chatbot-arena-leaderboard.hf.space/"

    # --- Fetch/Load Data ---
    # Hugging Face v2
    hg2_df = dt_if_missing("huggingface_v2.csv", fetch_hg_v2_leaderboard, hg_v2_url)

    # LMSYS
    lmsys_df = dt_if_missing("lmsys.csv", fetch_lmsys_leaderboard, lmsys_url)

    # --- Merge Data ---
    if hg2_df is not None and lmsys_df is not None:
         # Pass copies to merge_leaderboards to avoid modifying the original dataframes
         # especially important if dt_if_missing loaded them from cache
         merged_df = dt_if_missing("merged.csv", merge_leaderboards, hg2_df.copy(), lmsys_df.copy())
    else:
         logging.warning("Skipping merge because one or both source dataframes failed to load/fetch.")

    logging.info("Script finished.")
