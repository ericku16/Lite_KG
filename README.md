# Lite-KG (Lightweight Knowledge Graph Extractor)

This project is a **lightweight** and modular Knowledge Graph (KG) extraction pipeline, specialized for the **automotive supply chain**. It is designed to extract structured information from plain text (.txt) or PDF files and suitable for use with local LLMs via **Ollama**.

The pipeline is based on the 3-step process defined:
* **Step 1: Ontology Filtering** (using an LLM)
* **Step 2: NER + Entity Linking** (using Flair NER and the Wikidata API)
* **Step 3: Relation Extraction (NRE)** (using an LLM)

This project allows you to freely switch between LLM engines like OpenAI (e.g., GPT-4o) or local Ollama models (e.g., Llama 3, Mistral) using the "master switch" in `examples/run_extraction.py`.


## Please follow these steps to set up your environment.

## 1. Clone the Project
```bash
git clone [https://github.com/ericku16/Lite-KG.git]
cd lite-kg

# Create a new conda environment 
conda create -n lite-kg python=3.10

# Activate the environment
conda activate lite-kg

# Install all required packages from requirements.txt
pip install -r requirements.txt
```

###  Use Your Own Custom-Trained NER-Model
This project requires you to **provide your own custom-trained Flair NER model** (a `.pt` file). The `model/` directory is ignored by `.gitignore`.

1. Create a `model` folder in the project's **root directory** (at the same level as `src`).
2. Place your custom-trained `.pt` file (e.g., `my_ner_model.pt`) inside this `model` folder.
3. Open examples/run_extraction.py.
4. Change the `NER_MODEL_PATH` variable to match the name of your model file.


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
Put all the `.txt` or `.pdf` files you want to analyze into the `examples/example` folder.

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
## License
The MIT License.


