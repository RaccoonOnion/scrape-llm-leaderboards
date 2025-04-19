import os
import json
import pickle
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import traceback # Import traceback for detailed error logging

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
CSV_DIR       = "csv"
LIVEBENCH_CSV = os.path.join(CSV_DIR, "livebench_custom.csv")
LMSYS_CSV     = os.path.join(CSV_DIR, "lmsys.csv")
MAPPING_PATH  = "model_mapping_new.json" # File to store the model name mapping

# --- Model Configuration ---
# Define the OpenAI model to use for name matching
# *** Changed model from preview to standard gpt-4o-mini ***
# If this still fails, try "gpt-4o"
OPENAI_MODEL = "gpt-4o-mini" # Ensure this model is available and suitable

# --- Main Script Logic ---

# 1) Load your two leaderboards from CSV files
try:
    livebench = pd.read_csv(LIVEBENCH_CSV)
    lmsys      = pd.read_csv(LMSYS_CSV)
    print(f"Successfully loaded data from {LIVEBENCH_CSV} and {LMSYS_CSV}")
except FileNotFoundError as e:
    print(f"Error loading CSV files: {e}. Make sure '{CSV_DIR}' directory exists and contains the files.")
    exit(1) # Exit with non-zero code on error
except Exception as e:
    print(f"An unexpected error occurred while loading CSV files: {e}")
    exit(1)

# Initialize mapping variable
mapping = None

# 2) Try to load an existing mapping file
if os.path.exists(MAPPING_PATH):
    try:
        with open(MAPPING_PATH, "r") as f:
            mapping_content = f.read()
            # Check if the file is empty before trying to load JSON
            if not mapping_content.strip():
                 print(f"Warning: Mapping file '{MAPPING_PATH}' is empty. Will attempt to generate a new one.")
                 mapping = None # Treat as if file doesn't exist
            else:
                mapping = json.loads(mapping_content)
                # — if it’s accidentally a JSON‑string within a string, parse it again —
                if isinstance(mapping, str):
                    print("Detected mapping stored as a JSON string, attempting re-parse...")
                    mapping = json.loads(mapping)
                    # overwrite with the corrected structure
                    print("Re-saving mapping with corrected structure.")
                    with open(MAPPING_PATH, "w") as f_write:
                        json.dump(mapping, f_write, indent=2)
                # Validate that mapping is a list
                if isinstance(mapping, list):
                    print(f"Loaded existing mapping ({len(mapping)} entries) from {MAPPING_PATH}")
                else:
                    print(f"Error: Content in {MAPPING_PATH} is not a valid JSON list. Will attempt to generate a new one.")
                    mapping = None # Reset mapping if content is not a list
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from existing mapping file '{MAPPING_PATH}': {e}. Will attempt to generate a new one.")
        mapping = None # Reset mapping if file is corrupt
    except Exception as e:
        print(f"An unexpected error occurred while loading '{MAPPING_PATH}': {e}")
        traceback.print_exc() # Print traceback for unexpected errors
        mapping = None

# 3) Generate mapping if not loaded
if mapping is None:
    print("Generating new model mapping using OpenAI API...")
    # Extract the unique model‐name lists, handling potential missing columns
    try:
        # Ensure the columns exist before proceeding
        if "Model" not in livebench.columns:
             raise KeyError("'Model' column not found in livebench CSV.")
        if "model" not in lmsys.columns:
             raise KeyError("'model' column not found in lmsys CSV.")

        names_lb = livebench["Model"].dropna().unique().tolist()
        names_ls = lmsys["model"].dropna().unique().tolist()
    except KeyError as e:
        print(f"Error accessing columns: {e}. Please check CSV file headers.")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while extracting model names: {e}")
        exit(1)


    # Check if lists are empty
    if not names_lb or not names_ls:
        print("Error: One or both model name lists are empty after loading and cleaning. Cannot generate mapping.")
        exit(1)

    # 4) Craft the LLM prompt
    prompt = f"""
You are given two lists of AI model names scraped from different sites:

List A (livebench):
{names_lb}

List B (lmsys):
{names_ls}

Carefully compare the names. For each name in List A, find the most likely corresponding name in List B.
- If you find a plausible match, include it. Use your best judgment for variations in naming (e.g., 'GPT-4' vs 'gpt-4-0314').
- If no reasonable match exists in List B for a name in List A, set the value for 'model_lmsys' to null for that entry.
- Include a 'confidence' score (a number between 0.0 and 1.0) indicating how certain you are about the match (1.0 for exact or near-exact, lower for plausible but less certain matches, 0.0 or low if setting to null).

Output ONLY a valid JSON array of objects. Each object must have these exact keys:
  - "model_livebench" (string): The model name from List A.
  - "model_lmsys" (string or null): The corresponding model name from List B, or null if no match.
  - "confidence" (number): Your confidence score for the match (0.0 to 1.0).

Example format:
[
  {{
    "model_livebench": "Example Model A1",
    "model_lmsys": "example-model-a-v1",
    "confidence": 0.9
  }},
  {{
    "model_livebench": "Another Model A2",
    "model_lmsys": null,
    "confidence": 0.1
  }}
]

Respond with ONLY the raw JSON array, starting with '[' and ending with ']'. Do not include any introductory text, explanations, or markdown formatting like ```json.
"""

    # 5) Call the LLM
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: OPENAI_API_KEY environment variable not set.")
            exit(1)

        client = OpenAI(api_key=api_key)
        print(f"Sending request to OpenAI model: {OPENAI_MODEL}...")
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert AI assistant specialized in matching potentially inconsistent AI model names. You output ONLY valid JSON arrays according to the user's schema, without any extra text or markdown."},
                {"role": "user",   "content": prompt}
            ],
            # *** Increased max_tokens to prevent response truncation ***
            max_tokens=8192, # Adjust if necessary based on list sizes and model limits
            # temperature=0.0 # Consider adding for more deterministic JSON output if supported
        )

        # Check if response structure is as expected
        if resp.choices and resp.choices[0].message and resp.choices[0].message.content:
            raw = resp.choices[0].message.content.strip() # Use strip()

            # --- Debugging: Print the raw content string ---
            print(f"--- Raw content received from LLM ---\n{raw}\n--- End Raw content ---")

            # *** Clean potential markdown fences ***
            if raw.startswith("```json"):
                raw = raw[len("```json"):].strip()
                print("Removed leading ```json marker.")
            if raw.endswith("```"):
                raw = raw[:-len("```")].strip()
                print("Removed trailing ``` marker.")
            # **************************************

            # 5.1) Parse the JSON mapping
            if not raw:
                print("Error: Received empty content string from the LLM after cleaning. Cannot parse JSON.")
                mapping = [] # Use an empty list
            else:
                try:
                    # Attempt to find the start of the JSON array '[' or object '{'
                    json_start = raw.find('[')
                    if json_start == -1:
                         json_start = raw.find('{')

                    if json_start == 0: # JSON should start at the beginning after cleaning
                        print(f"--- Attempting to parse cleaned JSON content ---")
                        mapping = json.loads(raw) # Parse the cleaned raw string
                        # Validate that the parsed result is a list
                        if isinstance(mapping, list):
                             print(f"Successfully parsed JSON list from LLM response ({len(mapping)} items).")
                        else:
                             print(f"Error: Parsed JSON is not a list (type: {type(mapping)}). Content: {raw}")
                             mapping = [] # Reset if not a list
                    elif json_start > 0:
                         # This case indicates unexpected leading characters even after cleaning
                         print(f"Warning: JSON seems to start after some initial characters (index {json_start}) even after cleaning. Attempting parse from there.")
                         print(f"Initial characters: {raw[:json_start]}")
                         try:
                              mapping = json.loads(raw[json_start:])
                              if isinstance(mapping, list):
                                   print(f"Successfully parsed JSON list from index {json_start} ({len(mapping)} items).")
                              else:
                                   print(f"Error: Parsed JSON from index {json_start} is not a list (type: {type(mapping)}).")
                                   mapping = []
                         except json.JSONDecodeError as e_inner:
                              print(f"Error decoding JSON starting from index {json_start}: {e_inner}")
                              mapping = []
                    else:
                        # JSON start character not found
                        print("Error: Could not find the start of JSON ('[' or '{') in the cleaned LLM response.")
                        print("Cleaned LLM response content was:")
                        print(raw)
                        mapping = [] # Use an empty list

                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from cleaned LLM response: {e}")
                    print("Cleaned LLM response content was:")
                    print(raw)
                    mapping = [] # Use an empty list
        else:
            # Handle issues with the response structure itself (e.g., no choices, safety block)
            print("Error: LLM response structure was unexpected, choices or content was missing.")
            # Log the reason if available (e.g., finish_reason)
            finish_reason = resp.choices[0].finish_reason if resp.choices else "unknown"
            print(f"Finish Reason: {finish_reason}")
            print(f"Full Response object: {resp}")
            mapping = [] # Use an empty list

    except Exception as e:
        print(f"An error occurred during the OpenAI API call or processing: {e}")
        traceback.print_exc() # Print full traceback for API errors
        mapping = [] # Use an empty list

    # 6) Save the generated mapping (if successfully parsed as a non-empty list)
    if isinstance(mapping, list) and mapping:
        try:
            with open(MAPPING_PATH, "w") as f:
                json.dump(mapping, f, indent=2)
            print(f"Generated and saved new mapping ({len(mapping)} entries) to {MAPPING_PATH}")
        except Exception as e:
             print(f"Error saving the newly generated mapping to {MAPPING_PATH}: {e}")
             traceback.print_exc()
    elif isinstance(mapping, list) and not mapping:
         print("Warning: Mapping generation resulted in an empty list (possibly due to parsing errors or no matches found by LLM). No mapping file saved.")
         # Decide if the script should continue or exit based on whether an empty mapping is acceptable
         # exit(1) # Uncomment to stop if an empty mapping is an error
    else:
        # This case should ideally not be reached if error handling above is correct
        print("Error: Mapping generation failed or produced invalid type. Cannot save mapping.")
        mapping = [] # Ensure mapping is an empty list for downstream safety


# Ensure mapping is a list before proceeding (it should be [] if any error occurred)
if not isinstance(mapping, list):
     print("Critical Error: 'mapping' variable is not a list before merge step. Exiting.")
     exit(1)

# 7) Build the pandas‐friendly dict for merging
# Use .get() for safer access to potentially missing keys in the JSON items
map_dict = {}
try:
    map_dict = {
        item.get("model_livebench"): item.get("model_lmsys")
        for item in mapping # Iterate through the list
        # Ensure item is a dictionary and both keys exist before checking the lmsys value
        if isinstance(item, dict) and "model_livebench" in item and "model_lmsys" in item and item["model_lmsys"] is not None
    }
except Exception as e:
    print(f"An error occurred while building the mapping dictionary: {e}")
    traceback.print_exc()
    # map_dict remains empty or partially filled, proceed with caution


if not map_dict and mapping: # Only warn if mapping list was populated but yielded no valid pairs for the dict
    print("Warning: The generated mapping did not contain any valid pairs with non-null 'model_lmsys'. The merge might not link many rows.")
elif not mapping:
     print("Proceeding without any model name mapping (mapping list is empty).")
else:
     print(f"Built mapping dictionary with {len(map_dict)} entries.")


# 8) Standardize model names in livebench using the map and merge
try:
    # Add the 'model_std' column; missing models in map_dict will result in NaN
    # *** Explicitly cast to object dtype to prevent float64 issues with merge ***
    livebench["model_std"] = livebench["Model"].map(map_dict).astype(object)
    print("Created 'model_std' column in livebench DataFrame.")
    # Optional: Print dtypes to verify
    # print("Livebench dtypes after adding model_std:\n", livebench.dtypes)
    # print("LMSys dtypes:\n", lmsys.dtypes)


    # Perform the merge
    print("Attempting to merge DataFrames...")
    merged = livebench.merge(
        lmsys,
        left_on="model_std",      # Use the standardized name from livebench (object dtype)
        right_on="model",         # Match with the 'model' column in lmsys (should be object/string)
        how="left",               # Keep only keys that present in both livebench and lmsys
        suffixes=("_lb", "_ls")   # Suffixes for overlapping column names
    )
    print(f"Successfully merged the two dataframes. Result shape: {merged.shape}")

    # 9) Save the merged table
    output_csv = "merged_leaderboards.csv"
    output_pkl = "merged_leaderboards.pkl"
    merged.to_csv(output_csv, index=False)
    print(f"✅ Merged results saved to {output_csv}")

    # (Optional) pickle for faster reloads
    # merged.to_pickle(output_pkl)
    # print(f"✅ Pickled merged results to {output_pkl}")

except KeyError as e:
    print(f"Error during merge/save: Missing column {e}. Check CSV headers and mapping keys.")
    traceback.print_exc()
except ValueError as e:
    # Catch potential merge errors like the dtype mismatch more specifically
    print(f"Error during merge operation: {e}")
    print("This might indicate a persistent dtype mismatch or other merge issue.")
    # Check dtypes only if the columns exist
    if 'model_std' in livebench.columns:
        print("Livebench 'model_std' dtype:", livebench['model_std'].dtype)
    if 'model' in lmsys.columns:
        print("LMSys 'model' dtype:", lmsys['model'].dtype)
    traceback.print_exc()
except Exception as e:
    print(f"An unexpected error occurred during merging or saving results: {e}")
    traceback.print_exc()
