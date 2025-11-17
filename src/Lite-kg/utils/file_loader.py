import os
import fitz  # PyMuPDF

def load_document(file_path: str) -> str:
    """
    根據副檔名 (.txt 或 .pdf) 讀取檔案內容。
    """
    content = ""
    try:
        if file_path.endswith(".pdf"):
            with fitz.open(file_path) as doc:
                content = "".join(page.get_text() for page in doc)
        elif file_path.endswith(".txt"):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        else:
            print(f"  - Warning: Skipping unsupported file type: {file_path}")
            
    except Exception as e:
        print(f"  ❌ Error reading file {file_path}: {e}")
        
    return content