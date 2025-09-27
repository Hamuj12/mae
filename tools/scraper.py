import time
import json
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

# Setup headless Chrome
chrome_options = Options()
chrome_options.add_argument("--headless=new")
driver = webdriver.Chrome(options=chrome_options)

url = "https://www.blenderkit.com/?query=category_subtree:hdr-outdoor+order:-created+availability:free+trueHDR:true"
driver.get(url)

# Scroll until all assets are loaded
SCROLL_PAUSE_TIME = 1
asset_count = 0
last_height = driver.execute_script("return document.body.scrollHeight")

while True:
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(SCROLL_PAUSE_TIME)  # obey crawl-delay

    new_height = driver.execute_script("return document.body.scrollHeight")

    # Count current loaded assets
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, "html.parser")
    anchors = soup.find_all("a", class_="asset-image")
    asset_count = len(anchors)
    print(f"Loaded {asset_count} assets so far...")

    if asset_count >= 1919:  # stop if we got all assets
        break
    if new_height == last_height:  # no more content to load
        break
    last_height = new_height

# Extract IDs into strings
results = []
for a in anchors:
    href = a.get("href", "")
    if "/asset-gallery-detail/" in href:
        asset_id = href.split("/asset-gallery-detail/")[1].split("/")[0]
        entry = f"asset_base_id:{asset_id} asset_type:hdr"
        results.append(entry)

# Save to JSON array of strings
with open("assets.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved {len(results)} assets to assets.json")

driver.quit()