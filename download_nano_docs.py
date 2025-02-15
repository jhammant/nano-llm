import requests
import json
import difflib
import os

# GitHub Repo & Docs Directory
REPO_OWNER = "nanocurrency"
REPO_NAME = "nano-docs"
DOCS_URL = "https://docs.nano.org/"
OUTPUT_FILE = "nano_docs.jsonl"

# GitHub API URL (root directory of nano-docs)
GITHUB_API_URL = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/"

# Load valid URLs from `nano_urls.json`
with open("nano_urls.json", "r") as f:
    VALID_DOC_URLS = json.load(f)

# Optional GitHub Token (for rate limit bypassing)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise ValueError("❌ ERROR: GITHUB_TOKEN not found. Set it as an environment variable.")
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}

# Find the closest matching URL for a given filename
def find_best_doc_url(doc_name):
    best_match = difflib.get_close_matches(f"{DOCS_URL}{doc_name}/", VALID_DOC_URLS, n=1, cutoff=0.4)
    return best_match[0] if best_match else DOCS_URL  # Default to homepage if no match found

# Fetch all Markdown files recursively
def fetch_docs(url):
    response = requests.get(url, headers=HEADERS)

    if response.status_code == 404:
        print("❌ Error: Repository or directory not found. Check API URL.")
        return []

    try:
        files = response.json()
    except json.JSONDecodeError:
        print("❌ Error: Invalid JSON response from GitHub API")
        print("Response:", response.text)
        return []

    if not isinstance(files, list):
        print("❌ Unexpected response format from GitHub API")
        print("Response:", response.text)
        return []

    md_files = []
    for file in files:
        if "download_url" in file and file["name"].endswith(".md"):
            md_files.append(file)
        elif "type" in file and file["type"] == "dir":  # If it's a directory, fetch its contents
            md_files.extend(fetch_docs(file["url"]))

    return md_files

# Process and Convert Docs to JSONL
def process_docs():
    files = fetch_docs(GITHUB_API_URL)
    dataset = []

    for file in files:
        content_url = file["download_url"]
        doc_name = file["name"].replace(".md", "")

        # Find the best matching valid URL
        doc_url = find_best_doc_url(doc_name)

        content_response = requests.get(content_url, headers=HEADERS)

        if content_response.status_code == 200:
            content = content_response.text
        else:
            print(f"❌ Skipping {file['name']} due to failed download. Status: {content_response.status_code}")
            continue

        entry = {
            "messages": [
                {"role": "system", "content": "You are a Nano cryptocurrency expert. Always provide documentation links where possible."},
                {"role": "user", "content": f"What is {doc_name}?"},
                {"role": "assistant", "content": f"{content}\n\nFor more details, see: {doc_url}"}
            ]
        }
        dataset.append(entry)

    with open(OUTPUT_FILE, "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")

    print(f"✅ Dataset saved as {OUTPUT_FILE}, with {len(dataset)} entries.")

# Run the script
process_docs()
