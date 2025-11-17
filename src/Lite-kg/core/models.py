from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class EntityDetail(BaseModel):
    """
    Detailed information used to store entities, including standard names and Wikidata IDs.
    """
    name: str                 # LLM RE Step Return Name
    canonical_name: str       # Standard name from Wikidata
    wikidata_id: Optional[str] = None

class Triple(BaseModel):
    """
    The final stored knowledge triple structure
    """
    subject: EntityDetail
    predicate: str
    object: EntityDetail

class NERLinkResult(BaseModel):
    """
    Used to transfer data between Step 2 (ner_linker) and Step 3 (extractor)
    """
    # Key: Lowercase version of the original name or lowercase version of the standardized name
    # Value: A dictionary containing canonical_name and wikidata_id
    lookup_map: Dict[str, Dict[str, Any]]
    
    # A list of standardized entity names used for NRE hints
    # (e.g., ["TSMC", "Apple Inc."])
    canonical_entities_for_re: List[str]