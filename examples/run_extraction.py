import os
import sys
from dotenv import load_dotenv

# --- Step: Add src directory to Python path ---
# Import 'Extractor' from 'src/light_kg'
script_dir = os.path.dirname(__file__)

# Get project root (one level up from 'examples')
project_root = os.path.abspath(os.path.join(script_dir, '..'))

# Get the path to the 'src' folder
src_root = os.path.join(project_root, 'src')

# Add the 'src' folder to the front of the Python search path
sys.path.insert(0, src_root)

try:
    # Import Extractor
    from litekg.extractor import Extractor   # type: ignore[reportMissingImports]
except ImportError:
    print("Error: Could not import the 'litekg' module.")
    print(f"Please ensure the 'src' directory is located at: {src_root}")
    print("And that the 'litekg' folder exists within 'src'.")
    sys.exit(1)

# ==============================================================================
# --- Main Configuration ---
# ==============================================================================

# 1. Select LLM Provider ("openai" or "ollama")
PROVIDER = "ollama" 

# 2. Select Model based on Provider (e.g., "gpt-4o" or "mistral:latest") 
MODEL_NAME = "gpt-4o" if PROVIDER == "openai" else "mistral:latest"

# 3. Set NER Path
NER_MODEL_PATH = os.path.join(project_root, "model", "your_model.pt")

# 4. Set Folder Paths 
DOCUMENTS_FOLDER = os.path.join(script_dir, "example")
OUTPUT_FOLDER = os.path.join(script_dir, "output")

# 5. Other Settings 
CHUNK_SIZE = 5000
DELAY_BETWEEN_CHUNKS = 3 if PROVIDER == "openai" else 0

# ==============================================================================
# --- Main Execution Function ---
# ==============================================================================

def main():
    
    # --- Load API Key ---
    load_dotenv(os.path.join(project_root, '.env'))
    api_key = os.getenv("OPENAI_API_KEY")

    if PROVIDER == "openai" and not api_key:
        print("Error: 'openai' provider requires OPENAI_API_KEY. Please set it in your .env file.")
        return

    # --- Check Paths ---
    if not os.path.exists(NER_MODEL_PATH):
        print(f"Error: NER model path does not exist: {NER_MODEL_PATH}")
        print(f"Please update the 'NER_MODEL_PATH' variable in 'examples/run_extraction.py")
        print(f"The model file should be located at: {os.path.join(project_root, "model", "your_model.pt")}")
        return

    if not os.path.exists(DOCUMENTS_FOLDER):
        print(f"Documents folder '{DOCUMENTS_FOLDER}' not found.")
        print("Creating the folder for you. Please add .txt or .pdf files before running.")
        os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)
        return

    # --- Ensure output folder exists ---
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # --- Dynamically set output filename ---
    output_filename = f"result_{PROVIDER}_{MODEL_NAME.replace(':', '_')}.json"
    output_json_path = os.path.join(OUTPUT_FOLDER, output_filename)

    # --- 1. Initialize Extractor ---
    print(f"Initializing Extractor (Model: {PROVIDER} / {MODEL_NAME})")
    try:
        extractor = Extractor(
            provider=PROVIDER,
            model_name=MODEL_NAME,
            ner_model_path=NER_MODEL_PATH,
            api_key=api_key
        )
    except Exception as e:
        print(f"Extractor initialization failed: {e}")
        return

    # --- 2. Start Processing ---
    extractor.process_documents(
        folder_path=DOCUMENTS_FOLDER,
        output_json_path=output_json_path,
        chunk_size=CHUNK_SIZE,
        delay_between_chunks=DELAY_BETWEEN_CHUNKS
    )

    print(f"\n Execution complete! Results saved to:\n{output_json_path}")

if __name__ == "__main__":
    main()