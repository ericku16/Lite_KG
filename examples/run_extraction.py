import os
import sys
from dotenv import load_dotenv

# --- 步驟：將 src 目錄添加到 Python 路徑中 ---
# 從 'src/light_kg' 導入 'Extractor'
script_dir = os.path.dirname(__file__)

# 取得專案根目錄，即 'examples' 的上一層
project_root = os.path.abspath(os.path.join(script_dir, '..'))

# 取得 'src' 資料夾的路徑
src_root = os.path.join(project_root, 'src')

# 將 'src' 資料夾添加到 Python 搜尋路徑的最前面
sys.path.insert(0, src_root)

try:
    # 導入 Extractor 
    from light_kg.extractor import Extractor
except ImportError:
    print("❌ 錯誤: 無法導入 'light_kg' 模組")
    print(f"  請確保 'src' 目錄位於: {src_root}")
    print("  並且 'src' 目錄下有 'light_kg' 資料夾")
    sys.exit(1)

# ==============================================================================
# --- 主要設定  ---
# ==============================================================================

# 1. 選擇 LLM Provider: "openai" 或 "ollama"
PROVIDER = "openai" 

# 2. 根據 Provider 選擇模型
MODEL_NAME = "gpt-4o" if PROVIDER == "openai" else "mistral:latest"

# 3. 設定 NER 路徑
NER_MODEL_PATH = os.path.join(project_root, "model", "ner_model", "final-model.pt")

# 4. 設定資料夾路徑 
DOCUMENTS_FOLDER = os.path.join(script_dir, "example")
OUTPUT_FOLDER = os.path.join(script_dir, "output")

# 5. 其他設定 
CHUNK_SIZE = 5000
DELAY_BETWEEN_CHUNKS = 3 if PROVIDER == "openai" else 0

# ==============================================================================
# --- 主執行函式 ---
# ==============================================================================

def main():
    
    # --- 載入 API 金鑰 ---
    load_dotenv(os.path.join(project_root, '.env'))
    api_key = os.getenv("OPENAI_API_KEY")

    if PROVIDER == "openai" and not api_key:
        print("❌ 錯誤: 'openai' provider 需要 OPENAI_API_KEY，請在 .env 檔案中設定")
        return

    # --- 檢查路徑 ---
    if not os.path.exists(NER_MODEL_PATH):
        print(f"❌ 錯誤: NER 模型路徑不存在: {NER_MODEL_PATH}")
        print(f"  請在 'examples/run_extraction.py' 中更新 'NER_MODEL_PATH' 變數")
        print(f"  實際模型檔案應位於: {os.path.join(project_root, 'model', 'ner_model', 'final-model.pt')}")
        return

    if not os.path.exists(DOCUMENTS_FOLDER):
        print(f"⚠️ 警告: 文件資料夾 '{DOCUMENTS_FOLDER}' 不存在")
        print("  正在為您建立資料夾，請在執行前放入 .txt 或 .pdf 檔案")
        os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)
        return

    # --- 確保輸出資料夾存在 ---
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # --- 動態設定輸出檔案名稱 ---
    output_filename = f"result_{PROVIDER}_{MODEL_NAME.replace(':', '_')}.json"
    output_json_path = os.path.join(OUTPUT_FOLDER, output_filename)

    # --- 1. 初始化 Extractor ---
    print(f"--- 正在初始化 Extractor (模型: {PROVIDER} / {MODEL_NAME}) ---")
    try:
        extractor = Extractor(
            provider=PROVIDER,
            model_name=MODEL_NAME,
            ner_model_path=NER_MODEL_PATH,
            api_key=api_key
        )
    except Exception as e:
        print(f"❌ 初始化 Extractor 失敗: {e}")
        return

    # --- 2. 執行處理 ---
    extractor.process_documents(
        folder_path=DOCUMENTS_FOLDER,
        output_json_path=output_json_path,
        chunk_size=CHUNK_SIZE,
        delay_between_chunks=DELAY_BETWEEN_CHUNKS
    )

    print(f"\n🎉 執行完畢！結果已儲存至:\n{output_json_path}")

# --- 程式執行入口 ---
if __name__ == "__main__":
    main()