import csv
from selenium import webdriver
from bs4 import BeautifulSoup
import os

# --- Configuration ---
CSV_DIR = "csv"
LIVEBENCH_URL = "https://livebench.ai/#/"
OUTPUT_CSV = os.path.join(CSV_DIR, "livebench.csv")

# Start the browser
driver = webdriver.Chrome()  # Make sure chromedriver is installed & in PATH
driver.get(LIVEBENCH_URL)
driver.implicitly_wait(5)  # Wait for JavaScript to load content

# Parse page content with BeautifulSoup
html = driver.page_source
soup = BeautifulSoup(html, "html.parser")

# Extract headers
headers = [th.get_text(strip=True) for th in soup.select("table.main-tabl thead th")]
headers.append("Model Link")  # Add extra column for the model's href

# Extract rows
data_rows = []
rows = soup.select("table.main-tabl tbody tr")
for row in rows:
    cells = row.select("td")
    row_data = [cell.get_text(strip=True) for cell in cells]

    # Get model link
    link = row.select_one("td.sticky-col.model-col a")
    row_data.append(link["href"] if link else "")

    data_rows.append(row_data)

driver.quit()

os.makedirs(CSV_DIR, exist_ok=True) # Ensure dir exists

# Save to CSV
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(data_rows)

print(f"✅ CSV saved as {OUTPUT_CSV}")