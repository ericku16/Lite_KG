import time
import requests
from flair.data import Sentence
from flair.models import SequenceTagger
from typing import List, Dict, Any
from ..core.models import NERLinkResult

class NERLinker:
    """
    [Step 2 & 2.5] 執行 Flair NER 並使用 Wikidata API 進行實體連結

    """

    def __init__(self, ner_model_path: str):
        self.tagger = self._load_ner_model(ner_model_path)
        self.wikidata_headers = {'User-Agent': 'FlexiKG/1.0 (your_email@example.com)'}
        self.wikidata_api_endpoint = "https://www.wikidata.org/w/api.php"
        self.total_linking_time = 0.0

    def _load_ner_model(self, model_path: str):
        """載入 Flair NER 模型。"""
        try:
            tagger = SequenceTagger.load(model_path)
            print(f"✅ Flair NER 模型從 '{model_path}' 載入成功！")
            return tagger
        except Exception as e:
            print(f"❌ 載入 Flair NER 模型失敗: {e}")
            return None

    def _get_ner_entities(self, text: str) -> List[str]:
        """
        [Step 2] 使用 Flair NER 模型從 ontology filter 後的文本中提取實體
        Input: shorter texts from step 1 
        Output: entities
        """
        if not self.tagger:
            print("  - Step 2 (NER): NER 模型未載入，跳過")
            return []
            
        sentence = Sentence(text)
        self.tagger.predict(sentence)
        entities = sentence.get_spans('ner')
        
        raw_entity_texts = [entity.text for entity in entities]
        if len(raw_entity_texts) < 2:
            print(f"  - Step 2 (NER): 找到少於 2 個實體，跳過")
            return []
            
        print(f"  - Step 2 (NER): 找到 {len(raw_entity_texts)} 個原始實體: {raw_entity_texts[:5]}...")
        return raw_entity_texts

    def _get_wikidata_info_from_api(self, entity_name: str) -> Dict[str, Any]:
        """
        [Step 2.5] 使用官方 Wikidata API 和計分排序
        """
        params = {
            'action': 'wbsearchentities',
            'format': 'json',
            'language': 'en',
            'search': entity_name,
            'limit': 5
        }

        try:
            response = requests.get(self.wikidata_api_endpoint, params=params, headers=self.wikidata_headers)
            response.raise_for_status()
            data = response.json()

            if not data.get("search"):
                print(f"    - API 未能找到 '{entity_name}' 的任何候選")
                return None

            candidates = data["search"]
            best_candidate = None
            highest_score = -1
            search_term_lower = entity_name.lower()

            for candidate in candidates:
                score = 0
                label = candidate.get("label", "")
                description = candidate.get("description", "").lower()
                aliases = [alias.lower() for alias in candidate.get("aliases", [])]

                if label.lower() == search_term_lower or search_term_lower in aliases:
                    score += 100
                elif search_term_lower in label.lower():
                    score += 50
                
                if any(kw in description for kw in ["company", "manufacturer", "brand", "supplier", "city", "country"]):
                    score += 20
                if any(kw in description for kw in ["person", "album", "genus"]):
                    score -= 50

                if score > highest_score:
                    highest_score = score
                    best_candidate = candidate
            
            if best_candidate and highest_score >= 20:
                q_id = best_candidate.get("id")
                final_label = best_candidate.get("label")
                desc = best_candidate.get("description", "N/A")
                print(f"    - API 找到 '{entity_name}' 的最佳匹配 (分數: {highest_score}): {q_id} - {final_label} ({desc})")
                return {
                    'name': entity_name,
                    "canonical_name": final_label,
                    "wikidata_id": q_id
                }
            else:
                print(f"    - API 未找到 '{entity_name}' 的【高可信度】候選")
                return None

        except requests.exceptions.RequestException as e:
            print(f"    - 呼叫 Wikidata API 時發生網路錯誤: {e}")
            return None

    def link_entities(self, text: str) -> NERLinkResult:
        """
        [Step 2 & 2.5] 執行 NER 並進行實體連結。
        """
        # Step 2: NER
        raw_entity_texts = self._get_ner_entities(text)
        if not raw_entity_texts:
            return NERLinkResult(lookup_map={}, canonical_entities_for_re=[])

        # Step 2.5: Entity Linking
        print("  - Step 2.5 (Linking): 正在使用 Wikidata API 進行實體連結...")
        linking_start = time.time()
        
        entity_map = {}
        unique_entities = sorted(list(set(raw_entity_texts)), key=str.lower)

        for entity_name in unique_entities:
            api_result = self._get_wikidata_info_from_api(entity_name)
            
            if api_result:
                entity_map[entity_name] = api_result
            else:
                # 即使連結失敗，也保留原始名稱作為標準名稱
                # 保持結構一致，包含 name 鍵
                entity_map[entity_name] = {
                    "name": entity_name,
                    "canonical_name": entity_name, 
                    "wikidata_id": None
                }
            
            time.sleep(0.1) 

        linking_end = time.time()
        linking_duration = linking_end - linking_start
        self.total_linking_time += linking_duration
        print(f"  - Step 2.5 (Linking): 本次連結耗時: {linking_duration:.2f} 秒")

        # 建立用於 RE 步驟的 lookup map 和標準化實體列表
        lookup = {}
        for raw_name, info in entity_map.items():
            if info and info.get('canonical_name'):
                lookup[raw_name.lower()] = info
                lookup[info['canonical_name'].lower()] = info
        
        # 直接從 entity_map 提取 canonical_name，避免重複處理
        canonical_entities_for_re = sorted(
            list(set(
                info['canonical_name'] 
                for info in entity_map.values() 
                if info and info.get('canonical_name')
            )),
            key=str.lower
        )
        
        print(f"  - Step 2.5 (Linking): 標準化後實體 ({len(canonical_entities_for_re)} 個): {canonical_entities_for_re[:5]}...")

        return NERLinkResult(
            lookup_map=lookup,
            canonical_entities_for_re=canonical_entities_for_re
        )