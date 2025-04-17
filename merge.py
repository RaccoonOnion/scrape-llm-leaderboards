import os
import pandas as pd
import logging
from typing import Optional, Any

# --- Configuration ---
CSV_DIR = "csv"
HG_CSV = os.path.join(CSV_DIR, "huggingface_v2.csv")
LMSYS_CSV = os.path.join(CSV_DIR, "lmsys.csv")
OUTPUT_CSV = os.path.join(CSV_DIR, "merged.csv")

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions (Optional but recommended for type consistency) ---

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


# --- Merging Function ---
def merge_leaderboards(df_hg: pd.DataFrame, df_lmsys: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Merges Hugging Face and LMSYS leaderboards on a derived key."""
    if df_hg is None or df_lmsys is None:
        logging.warning("One or both dataframes are None, cannot merge.")
        return None

    # Re-apply type setting after loading from CSV for consistency
    logging.info("Applying type conversions to loaded dataframes...")
    df_hg = set_column_types(df_hg.copy()) # Use copy to avoid SettingWithCopyWarning
    df_lmsys = set_column_types(df_lmsys.copy())

    # Create merge keys
    if 'model_name' in df_hg.columns:
        df_hg['key'] = df_hg.apply(lambda row: row['model_name'].split('/')[-1].lower().strip() if isinstance(row['model_name'], str) and '/' in row['model_name'] else str(row['model_name']).lower().strip(), axis=1)
    else:
        logging.error("Cannot create merge key for Hugging Face data: 'model_name' column missing.")
        return None

    if 'model_name' in df_lmsys.columns:
        df_lmsys['key'] = df_lmsys['model_name'].astype(str).str.lower().str.strip()
    else:
        logging.error("Cannot create merge key for LMSYS data: 'model_name' column missing.")
        return None

    # Find common keys for logging
    common_keys = set(df_hg['key'].dropna()) & set(df_lmsys['key'].dropna())
    logging.info(f"Found {len(common_keys)} common keys for merging.")
    if len(common_keys) < 10: logging.info(f"Sample common keys: {list(common_keys)[:10]}")

    # Perform the merge
    logging.info("Merging Hugging Face v2 and LMSYS data...")
    merged_df = pd.merge(df_hg, df_lmsys, on='key', how='inner', suffixes=('_hg', '_lmsys'))

    # Reorder columns
    key_cols = ['key', 'model_name_hg', 'model_url_hg', 'model_name_lmsys', 'model_url_lmsys']
    key_cols = [col for col in key_cols if col in merged_df.columns]
    other_cols = [col for col in merged_df.columns if col not in key_cols]
    merged_df = merged_df[key_cols + other_cols]

    logging.info(f"Successfully merged leaderboards. Shape: {merged_df.shape}")
    if merged_df.empty:
            logging.warning("Merged dataframe is empty. Check merge keys and source data.")
    return merged_df

# --- Main Execution ---
if __name__ == "__main__":
    logging.info("--- Starting Leaderboard Merger ---")
    os.makedirs(CSV_DIR, exist_ok=True) # Ensure dir exists

    # Check if input files exist
    if not os.path.exists(HG_CSV):
        logging.error(f"Input file not found: {HG_CSV}. Please run the Hugging Face scraper first.")
        exit() # Or handle differently
    if not os.path.exists(LMSYS_CSV):
        logging.error(f"Input file not found: {LMSYS_CSV}. Please run the LMSYS scraper first.")
        exit() # Or handle differently

    # Load data
    try:
        logging.info(f"Loading Hugging Face data from: {HG_CSV}")
        df_hg = pd.read_csv(HG_CSV)
        logging.info(f"Loading LMSYS data from: {LMSYS_CSV}")
        df_lmsys = pd.read_csv(LMSYS_CSV)
    except Exception as e:
        logging.error(f"Failed to load input CSV files: {e}")
        df_hg, df_lmsys = None, None # Ensure they are None if loading fails

    # Merge data
    if df_hg is not None and df_lmsys is not None:
        merged_df = merge_leaderboards(df_hg, df_lmsys)

        # Save merged data
        if merged_df is not None:
            try:
                merged_df.to_csv(OUTPUT_CSV, index=False)
                logging.info(f"Successfully saved merged data to: {OUTPUT_CSV}")
            except Exception as e:
                logging.error(f"Failed to save merged DataFrame to CSV {OUTPUT_CSV}: {e}")
        else:
            logging.warning("Merging function returned None. Cannot save merged CSV.")
    else:
        logging.warning("Skipping merge because one or both source dataframes failed to load.")

    logging.info("--- Leaderboard Merger Finished ---")
