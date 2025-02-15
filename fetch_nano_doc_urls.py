import requests
import json
import xml.etree.ElementTree as ET

SITEMAP_URL = "https://docs.nano.org/sitemap.xml"
OUTPUT_FILE = "nano_urls.json"

def fetch_sitemap_urls(sitemap_url):
    response = requests.get(sitemap_url)

    if response.status_code != 200:
        print("❌ Error: Could not fetch the sitemap.")
        return []

    # Parse XML response
    root = ET.fromstring(response.content)
    namespaces = {'s': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

    urls = []
    for url in root.findall("s:url", namespaces):
        loc = url.find("s:loc", namespaces)
        if loc is not None and loc.text.startswith("https://docs.nano.org/"):
            urls.append(loc.text)

    return urls

# Fetch all valid documentation URLs
valid_urls = fetch_sitemap_urls(SITEMAP_URL)

# Save to a file
with open(OUTPUT_FILE, "w") as f:
    json.dump(valid_urls, f, indent=4)

print(f"✅ Saved {len(valid_urls)} valid URLs to {OUTPUT_FILE}")
