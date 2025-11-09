import os
import json
import time
from typing import List

from .core.clients import get_llm_client, BaseLLMClient
from .core.models import Triple, EntityDetail
from .steps import OntologyFilter, NERLinker, RelationExtractor
from .utils.file_loader import load_document

class Extractor:
    """
    主要的知識圖譜提取器 
    這個類別初始化並協調所有步驟
    """

    def __init__(self, 
        provider: str,
        model_name: str, 
        ner_model_path: str, 
        api_key: str = None):

        """
        Args:
            provider (str): "openai" 或 "ollama"
            model_name (str): 模型的名稱 (例如 "gpt-4o", "mistral:latest")
            ner_model_path (str): 本地 Flair NER 模型的路徑
            api_key (str, optional): OpenAI API 金鑰
        """
        print(f"--- 正在初始化 Extractor ---")
        print(f"  Provider: {provider}")
        print(f"  LLM Model: {model_name}")
        print(f"  NER Model: {ner_model_path}")
        
        # 1. 初始化 LLM 客戶端 (用於 Step 1 和 Step 3)
        self.llm_client = get_llm_client(provider, model_name, api_key)
        
        # 2. 初始化所有步驟
        self.filterer = OntologyFilter(self.llm_client)
        self.ner_linker = NERLinker(ner_model_path)
        self.extractor = RelationExtractor(self.llm_client)
        print("--- 初始化完成 ---\n")

    def process_documents(
        self, 
        folder_path: str, 
        output_json_path: str, 
        chunk_size: int = 5000, 
        delay_between_chunks: int = 1
    ) -> List[Triple]:
        """
        處理資料夾中的所有文件並提取關係三元組
        """
        
        print(f"\n--- 開始處理資料夾 '{folder_path}' ---")
        output_dir = os.path.dirname(output_json_path)
        if not os.path.exists(output_dir): 
            os.makedirs(output_dir)
            
        all_extracted_triples: List[Triple] = []
        start_time = time.time()

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if not (os.path.isfile(file_path) and (filename.endswith(".txt") or filename.endswith(".pdf"))):
                continue
                
            print(f"\n📄 正在處理檔案: {filename}")
            
            # 使用工具函式讀取檔案
            content = load_document(file_path)
            if not content:
                print(f"  - 檔案內容為空 跳過 {filename}。")
                continue

            # --- 流程開始 ---
            try:
                # Step 1: Ontology Filtering
                filtered_content = self.filterer.filter_text(content)
                if not filtered_content or not filtered_content.strip():
                    print("  - 過濾後無相關內容 跳過此檔案")
                    continue

                # 切分文本
                text_chunks = [filtered_content[i:i + chunk_size] for i in range(0, len(filtered_content), chunk_size)]
                
                for i, chunk in enumerate(text_chunks):
                    print(f"\n  → 正在處理區塊 {i+1}/{len(text_chunks)}...")

                    # Step 2 & 2.5: NER + Entity Linking
                    # ner_link_result 包含 .lookup_map 和 .canonical_entities_for_re
                    ner_link_result = self.ner_linker.link_entities(chunk)
                    
                    if not ner_link_result.canonical_entities_for_re:
                        print("  - Step 2.5 (Linking): 未能找到足夠的實體 跳過此區塊")
                        continue

                    # Step 3: LLM-RE (NRE)
                    # 使用標準化後的實體清單進行關係提取
                    relations = self.extractor.extract_relations(
                        chunk, 
                        ner_link_result.canonical_entities_for_re
                    )
                    
                    if relations:
                        print(f"  - 正在進行最終格式化...")
                        formatted_count = 0
                        for rel in relations:
                            if isinstance(rel, list) and len(rel) == 3:
                                s, p, o = rel
                                
                                # 根據 RE 回傳的標準化名稱，從 lookup map 中查找 Wikidata ID
                                s_info = ner_link_result.lookup_map.get(s.lower())
                                o_info = ner_link_result.lookup_map.get(o.lower())
                                
                                s_canonical = s_info['canonical_name'] if s_info else s
                                s_id = s_info['wikidata_id'] if s_info else None
                                
                                o_canonical = o_info['canonical_name'] if o_info else o
                                o_id = o_info['wikidata_id'] if o_info else None
                                
                                # 使用 Pydantic 模型
                                final_triple = Triple(
                                    subject=EntityDetail(name=s, canonical_name=s_canonical, wikidata_id=s_id),
                                    predicate=p,
                                    object=EntityDetail(name=o, canonical_name=o_canonical, wikidata_id=o_id)
                                )
                                all_extracted_triples.append(final_triple)
                                formatted_count += 1

                        print(f"  ✅ 成功處理 {formatted_count} 筆關聯並加入 Wikidata ID ")

                    if i < len(text_chunks) - 1:
                        time.sleep(delay_between_chunks)
                        
            except Exception as e:
                print(f"  ❌ 處理檔案 {filename} 時發生嚴重錯誤：{e}")

        # --- 儲存結果 ---
        print(f"\n--- 正在將結果寫入 '{output_json_path}'... ---")
        
        # 將 Pydantic 模型轉換為字典列表以便 JSON 儲存
        results_data = [triple.model_dump() for triple in all_extracted_triples]
        
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"🚀 JSON 檔案產出成功！共提取 {len(all_extracted_triples)} 筆關聯")

        # --- 顯示總耗時 ---
        end_time = time.time()
        elapsed_time = end_time - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print("\n" + "="*50)
        print(f"⏱️  總耗時: {minutes} 分 {seconds} 秒 ({elapsed_time:.2f} 秒)")
        
        # 顯示實體連結總耗時
        linking_minutes = int(self.ner_linker.total_linking_time // 60)
        linking_seconds = int(self.ner_linker.total_linking_time % 60)
        print(f"🔗 實體連結 (API 查詢) 總耗時: {linking_minutes} 分 {linking_seconds} 秒 ({self.ner_linker.total_linking_time:.2f} 秒)")
        print("="*50)

        return all_extracted_triples