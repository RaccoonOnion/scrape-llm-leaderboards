import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException # Import exception for handling
from bs4 import BeautifulSoup
import os
import time

# --- Configuration ---
CSV_DIR = "csv"
LIVEBENCH_URL = "https://livebench.ai/#/"
OUTPUT_CSV = os.path.join(CSV_DIR, "livebench_custom.csv") # More generic name
WAIT_TIMEOUT = 10 # Seconds to wait for elements

# --- Interaction Configuration ---
# Define the interactions to perform before scraping.
# Each item is a dictionary specifying the locator strategy and selector.
# Add more dictionaries to this list to interact with more elements.
# Supported types: By.CSS_SELECTOR, By.ID, By.XPATH, By.CLASS_NAME, etc.
INTERACTIONS_CONFIG = [
    {
        "description": "Show API Name checkbox", # For logging/debugging
        "type": By.CSS_SELECTOR,
        "selector": "#showApiName"
    },
    # {
    #     "description": "Show Organization",
    #     "type": By.ID,
    #     "selector": "showProvider"
    # },
    # {
    #     "description": "Click a Button Example (if needed)",
    #     "type": By.XPATH,
    #     "selector": "//button[contains(text(), 'Apply Filters')]"
    # },
]

# How long to wait (in seconds) after performing ALL interactions,
# allowing time for JavaScript updates to the page/table. Adjust as needed.
POST_INTERACTION_WAIT = 3

# --- Main Script ---

# Start the browser
options = webdriver.ChromeOptions()
# options.add_argument('--headless')
# options.add_argument('--disable-gpu')
driver = webdriver.Chrome(options=options)
print(f"Navigating to {LIVEBENCH_URL}...")
driver.get(LIVEBENCH_URL)

try:
    # --- Interaction Loop ---
    wait = WebDriverWait(driver, WAIT_TIMEOUT)
    print("--- Performing Pre-Scrape Interactions ---")
    if not INTERACTIONS_CONFIG:
        print("No interactions configured.")
    else:
        for interaction in INTERACTIONS_CONFIG:
            desc = interaction.get("description", interaction["selector"]) # Get description or default to selector
            interaction_type = interaction["type"]
            interaction_selector = interaction["selector"]

            print(f"Attempting to click: '{desc}' (using {interaction_type}: {interaction_selector})")
            try:
                # Wait for the element to be clickable
                element = wait.until(EC.element_to_be_clickable((interaction_type, interaction_selector)))
                # Click the element
                element.click()
                print(f"  Successfully clicked '{desc}'.")
                # Optional: Add a very small pause between clicks if needed
                # time.sleep(0.5)
            except TimeoutException:
                print(f"  Warning: Element '{desc}' not found or not clickable within {WAIT_TIMEOUT} seconds.")
            except Exception as e:
                print(f"  Error clicking element '{desc}': {e}")

        # --- Wait after all interactions ---
        if INTERACTIONS_CONFIG and POST_INTERACTION_WAIT > 0:
            print(f"Waiting {POST_INTERACTION_WAIT} seconds for updates after interactions...")
            time.sleep(POST_INTERACTION_WAIT)
        print("--- Interactions Complete ---")

    # --- Scraping Step ---
    print("\nParsing page content with BeautifulSoup...")
    html = driver.page_source # Get the potentially updated HTML
    soup = BeautifulSoup(html, "html.parser")

    # Extract headers
    print("Extracting headers...")
    table_header = soup.select_one("table.main-tabl thead")
    if not table_header:
        print("Error: Could not find the table header.")
        # Optional debugging: save source
        # with open("debug_no_header.html", "w", encoding="utf-8") as f:
        #     f.write(driver.page_source)
        driver.quit()
        exit()

    headers = [th.get_text(strip=True) for th in table_header.select("th")]
    needs_model_link_column = "Model Link" not in headers
    if needs_model_link_column:
        headers.append("Model Link") # Add header if we plan to add the link as a separate column

    # Extract rows
    print("Extracting rows...")
    data_rows = []
    table_body = soup.select_one("table.main-tabl tbody")
    if not table_body:
        print("Warning: Could not find the table body or table is empty.")
    else:
        rows = table_body.select("tr")
        print(f"Found {len(rows)} rows.")
        for i, row in enumerate(rows):
            cells = row.select("td")
            row_data = [cell.get_text(strip=True) for cell in cells]

            # Get model link (adjust selector if needed based on table structure)
            link_element = row.select_one("td.sticky-col.model-col a")
            model_link = link_element["href"] if link_element and link_element.has_attr('href') else ""

            # Append the model link if a separate column was added
            if needs_model_link_column:
                 row_data.append(model_link)

            # Check for mismatch and pad if necessary
            if len(headers) != len(row_data):
                 print(f"Warning: Header count ({len(headers)}) doesn't match row {i} data count ({len(row_data)}). Row data: {row_data}")
                 # Simple padding - adjust if more sophisticated handling is needed
                 row_data.extend([""] * (len(headers) - len(row_data)))


            data_rows.append(row_data)

    os.makedirs(CSV_DIR, exist_ok=True)

    # Save to CSV
    print(f"\nSaving data to {OUTPUT_CSV}...")
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(data_rows)

    print(f"✅ CSV saved as {OUTPUT_CSV}")

except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")
    # Optional: Save page source on error for debugging
    # try:
    #     with open("error_page_source.html", "w", encoding="utf-8") as f:
    #         f.write(driver.page_source)
    #     print("Saved page source to error_page_source.html")
    # except Exception as e_save:
    #     print(f"Could not save page source on error: {e_save}")

finally:
    # --- Cleanup ---
    print("Closing the browser.")
    if 'driver' in locals() and driver:
        driver.quit()