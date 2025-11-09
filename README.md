# Light_KG (Lightweight Knowledge Graph Extractor)

This project is a modular Knowledge Graph (KG) extraction pipeline designed to extract structured information from plain text (.txt) or PDF files.

The pipeline is based on the 3-step process defined :
1.  **Step 1: Ontology Filtering** (using an LLM)
2.  **Step 2: NER + Entity Linking** (using Flair NER and the Wikidata API)
3.  **Step 3: Relation Extraction (NRE)** (using an LLM)

This project allows you to freely switch between LLM engines like OpenAI (e.g., GPT-4o) or local Ollama models (e.g., Llama 3, Mistral) using the "master switch" in `examples/run_extraction.py`.


## Please follow these steps to set up your environment.

```bash
##  1. Clone the Project
git clone [YOUR_GITHUB_PROJECT_URL]
cd light_kg

# Create a new conda environment (Python 3.10 or 3.11 is recommended)
conda create -n light_kg python=3.10

# Activate the environment
conda activate light_kg

# Install all required packages from requirements.txt
pip install -r requirements.txt

# The NER model (final-model.pt) is required for this project but is too large to be hosted on GitHub (it is ignored by .gitignore). You must download it manually.

# 1. [Download the model (final-model.pt) here]
# 2. After downloading, create a model folder in the project's root directory (at the same level as src).
# 3. Place the final-model.pt file inside this model folder.


## 2. Configuration
# Required if you plan to use the 'openai' provider
OPENAI_API_KEY="enter your api key"

# This is the default host if you are running Ollama locally.
# You only need to change this if your Ollama service is on a different machine.
OLLAMA_HOST="http://localhost:11434"

## 3. How to Execute
# Example: To use Ollama's mistral
PROVIDER = "ollama" 
MODEL_NAME = "mistral:latest"

# Example: To use OpenAI's GPT-4o
PROVIDER = "openai" 
MODEL_NAME = "gpt-4o"

## Run the Program: In your terminal (make sure your conda or venv environment is activated), run the following command:
python examples/run_extraction.py
