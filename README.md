# LLM Leaderboard Scraper and Merger

## Overview

This project scrapes data from multiple Large Language Model (LLM) leaderboards, namely [LiveBench.ai](https://livebench.ai/#/) and the [LMSYS Chatbot Arena Leaderboard](https://lmarena-ai-chatbot-arena-leaderboard.hf.space/). It utilizes an LLM (specifically configured to use OpenAI's `gpt-4o-mini` via API) to intelligently map potentially inconsistent model names between these sources. Finally, it merges the standardized data into a single CSV file for easier analysis and comparison.

## Features

* Scrapes leaderboard data from LiveBench.ai using Selenium and BeautifulSoup.
* Scrapes leaderboard data from the LMSYS Chatbot Arena Hugging Face Space by parsing embedded configuration data.
* Scrapes data from the Hugging Face Open LLM Leaderboard API (optional, separate script).
* Uses the OpenAI API to automatically generate mappings between model names found on different leaderboards.
* Merges data from LiveBench and LMSYS based on the generated mapping.
* Includes a validation script (`check_mapping.py`) to verify the integrity and coverage of the generated model name mapping.

## File Structure
```
scrape-llm-leaderboards
├─ API_merge.py
├─ LICENSE
├─ README.md
├─ check_mapping.py
├─ merged_leaderboards.csv
├─ model_mapping.json
├─ requirements.txt
├─ scrape_livebench.py
├─ scrape_lmsys.py
└─ scrape_open_llm_lb.py

```

## Setup

1.  **Clone/Download:** Get a copy of this repository.
2.  **Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **OpenAI API Key:**
    * This project requires an OpenAI API key to generate the model name mappings using the `API_merge.py` script.
    * Create a file named `.env` in the project's root directory.
    * Add your API key to the `.env` file like this:
        ```
        OPENAI_API_KEY=YOUR_API_KEY_HERE
        ```
    * The `API_merge.py` script uses `python-dotenv` to load this key.
5.  **WebDriver:**
    * The `scrape_livebench.py` script uses Selenium and requires a WebDriver.
    * Ensure you have Google Chrome installed.
    * Download the ChromeDriver executable that matches your Chrome version and place it in your system's PATH or specify its location in the script if needed.

## Usage

Follow these steps to scrape, map, and merge the leaderboard data:

1.  **Scrape LiveBench:** Run the script to get the latest data from LiveBench.ai.
    ```bash
    python scrape_livebench.py
    ```
    This will create/update `csv/livebench.csv`.

2.  **Scrape LMSYS:** Run the script to get the latest data from the LMSYS Arena leaderboard.
    ```bash
    python scrape_lmsys.py
    ```
    This will create/update `csv/lmsys.csv`.

3.  **Map and Merge Data:** Run the main merging script.
    ```bash
    python API_merge.py
    ```
    * This script first loads `csv/livebench.csv` and `csv/lmsys.csv`.
    * If `model_mapping.json` doesn't exist, is empty, or is invalid, it will call the OpenAI API to generate it. **Note:** This step requires a valid `OPENAI_API_KEY` in your `.env` file.
    * It then uses the mapping to merge the data and saves the result to `merged_leaderboards.csv`.

4.  **Validate Mapping (Optional):** Run the check script to ensure the mapping file is valid and covers the models found in the source CSVs.
    ```bash
    python check_mapping.py
    ```
    *(Note: The script might need adjustment if the mapping filename inside `check_mapping.py` differs from `model_mapping.json`)*.

5.  **Scrape Hugging Face Leaderboard (Optional):** Run this script if you also want data from the Hugging Face Open LLM Leaderboard API.
    ```bash
    python scrape_open_llm_lb.py
    ```
    This creates/updates `csv/huggingface_v2.csv`. This data is not used in the main merge process handled by `API_merge.py`.

## Dependencies

The project relies on the following Python libraries:

* `requests` 
* `pandas` 
* `beautifulsoup4` 
* `lxml` 
* `selenium` 
* `openai` (implicitly used by `API_merge.py`)
* `python-dotenv` (implicitly used by `API_merge.py`)

Install them using `pip install -r requirements.txt`.

## Configuration

* **URLs:** Leaderboard URLs are defined as constants near the top of `scrape_livebench.py`, `scrape_lmsys.py`, and `scrape_open_llm_lb.py`.
* **OpenAI Model:** The specific OpenAI model used for mapping is set in `API_merge.py` (variable `OPENAI_MODEL`, currently set to `gpt-4o-mini`).
* **LiveBench Interactions:** `scrape_livebench.py` contains an `INTERACTIONS_CONFIG` list where you can specify elements (like checkboxes or buttons) to interact with on the LiveBench page before scraping begins.

## Output

* **`merged_leaderboards.csv`:** The primary output file, containing merged data from LiveBench and LMSYS leaderboards, with model names standardized.
* **`model_mapping.json`:** The JSON file storing the mapping between LiveBench and LMSYS model names, generated (or loaded) by `API_merge.py`.
* **`csv/livebench.csv`**, **`csv/lmsys.csv`**, **`csv/huggingface_v2.csv`:** Raw data scraped from the respective sources.
