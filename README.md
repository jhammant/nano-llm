# Nano-Llama: Fine-Tuned LLM for Nano Cryptocurrency

## 🚀 Overview
This project fine-tunes the **DeepSeek LLM** model on Nano cryptocurrency documentation. It enables the model to answer Nano-related queries while linking to relevant documentation.

## 🛠️ Installation & Setup
### **1️⃣ Install Dependencies**
First, install Python packages:
```bash
pip install -r requirements.txt
```

### **2️⃣ Set Up GitHub Token (For Documentation Fetching)**
Since the script fetches documentation from GitHub, export your token **(DO NOT hardcode it)**:
```bash
export GITHUB_TOKEN="ghp_your_actual_token_here"
```
If you want it to persist:
```bash
echo 'export GITHUB_TOKEN="ghp_your_actual_token_here"' >> ~/.zshrc
source ~/.zshrc
```

---
## ⚡ Running the Scripts
### **1️⃣ Fetch Nano Documentation**
```bash
python3 download_nano_docs.py
```
This script downloads Nano-related markdown files from GitHub and processes them into a JSONL dataset.

### **2️⃣ Generate Documentation URLs**
```bash
python3 fetch_nano_doc_urls.py
```
This script scrapes `https://docs.nano.org/` to extract valid URLs for linking in responses.

### **3️⃣ Preprocess & Clean Dataset**
```bash
python3 clean_jsonl.py
```
This ensures the dataset is correctly formatted before fine-tuning.

### **4️⃣ Fine-Tune the Model**
```bash
accelerate launch fine_tune.py
```
This fine-tunes the model using **DeepSeek LLM** with the Nano documentation.

---
## 📂 Project Structure
```
📁 nano-llama
├── download_nano_docs.py   # Fetches Nano markdown docs from GitHub
├── fetch_nano_doc_urls.py  # Scrapes Nano documentation site URLs
├── clean_jsonl.py          # Cleans and formats the dataset
├── fine_tune.py            # Fine-tuning script using Hugging Face Trainer
├── requirements.txt        # Python dependencies
├── .gitignore              # Excludes model weights, cache, and logs
├── README.md               # This file
```

---
## 📜 Notes
- **Model weights are NOT included in the repo** to avoid large file commits.
- Fine-tuning is done on **Apple Silicon MPS (Mac GPU)**.
- To keep responses Nano-focused, system prompts & fine-tuning guide responses away from unrelated topics.

🚀 **Enjoy your Nano LLM!**
