# Light_KG (Lightweight Knowledge Graph Extractor)

This project is a **lightweight** and modular Knowledge Graph (KG) extraction pipeline, specialized for the **automotive supply chain**. It is designed to extract structured information from plain text (.txt) or PDF files and suitable for use with local LLMs via **Ollama**.

The pipeline is based on the 3-step process defined:
* **Step 1: Ontology Filtering** (using an LLM)
* **Step 2: NER + Entity Linking** (using Flair NER and the Wikidata API)
* **Step 3: Relation Extraction (NRE)** (using an LLM)

This project allows you to freely switch between LLM engines like OpenAI (e.g., GPT-4o) or local Ollama models (e.g., Llama 3, Mistral) using the "master switch" in `examples/run_extraction.py`.


## Please follow these steps to set up your environment.

## 1. Clone the Project
```bash
git clone [https://github.com/ericku16/Light_KG.git]
cd light_kg

# Create a new conda environment 
conda create -n light_kg python=3.10

# Activate the environment
conda activate light_kg

# Install all required packages from requirements.txt
pip install -r requirements.txt
```

### Option A: Use the Default NER-Model 
The default model (`final-model.pt`) is a custom-trained Flair NER model, **specialized for the automotive supply chain**.

This model is required for the project to run out-of-the-box but is too large to be hosted on GitHub (it is ignored by `.gitignore`). You must download it manually.

1. [Download the Default Model (final-model.pt) here](https://drive.google.com/drive/folders/18D1K_IQxwZPtPt7V6Ni_tOljvKazWfS9?usp=sharing)
2. After downloading, create a model folder in the project's root directory (at the same level as src).
3. Place the final-model.pt file inside this model folder.

### Option B: Use Your Own Custom-Trained NER-Model
This project can load any Flair-trained .pt model.

1. Place your own model (e.g., my_ner_model.pt) into the model/ folder.
2. Open examples/run_extraction.py.
3. Change the NER_MODEL_PATH variable to point to your new file's name.


## 2. Configuration
```bash
# Required if you plan to use the 'openai' provider
OPENAI_API_KEY="enter your api key"

# This is the default host if you are running Ollama locally.
# You only need to change this if your Ollama service is on a different machine.
OLLAMA_HOST="http://localhost:11434"
```

## 3. How to Execute

### Step 1: Place Your Files (there are some default examples)
Put all the `.txt` or `.pdf` files you want to analyze into the `examples` folder.

### Step 2: Select Your LLM
Open the file **`examples/run_extraction.py`** in your code editor.

At the very top, you will find the main settings. Edit the `PROVIDER` and `MODEL_NAME` variables to choose which LLM you want to use.

**Example: To use Ollama's Mistral**
```bash
# 1. Choose LLM Provider ("openai" or "ollama")
PROVIDER = "ollama" 

# 2. Choose Model Name (e.g., "gpt-4o" or "mistral:latest") 
MODEL_NAME = "mistral:latest"

# 3. Run the Program
python examples/run_extraction.py
```