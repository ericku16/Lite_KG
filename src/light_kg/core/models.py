from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class EntityDetail(BaseModel):
    """
    用於儲存實體的詳細資訊，包含標準名稱和 Wikidata ID。
    這對應您原始碼中的 subject_obj 和 object_obj 結構。
    """
    name: str                 # LLM RE 步驟回傳的名稱
    canonical_name: str       # 來自 Wikidata 的標準名稱
    wikidata_id: Optional[str] = None

class Triple(BaseModel):
    """
    最終儲存的知識三元組結構。
    這對應您原始碼中的 final_triple 結構。
    """
    subject: EntityDetail
    predicate: str
    object: EntityDetail

class NERLinkResult(BaseModel):
    """
    這是一個輔助模型 (data class)，
    用於在 Step 2 (ner_linker) 和 Step 3 (extractor) 之間傳遞資料。
    """
    # 鍵：原始名稱的小寫版本或標準化名稱的小寫版本
    # 值：包含 canonical_name 和 wikidata_id 的字典
    lookup_map: Dict[str, Dict[str, Any]]
    
    # 用於 NRE 提示的標準化實體名稱列表
    # (例如: ["TSMC", "Apple Inc."])
    canonical_entities_for_re: List[str]