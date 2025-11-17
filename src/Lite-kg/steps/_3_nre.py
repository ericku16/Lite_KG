import json
from ..core.clients import BaseLLMClient
from typing import List, Any

class RelationExtractor:
    """
    [Step 3] Use ollama to extract relation triples from the text after the ontology filter and entities extracted by NER.
    Input: shorter texts from step 1 and entities from step 2
    Output: triples
    """
    
    def __init__(self, llm_client: BaseLLMClient):
        self.llm_client = llm_client
        self.system_prompt = """You are an expert in Relation Extraction (RE). You will be given a text snippet and a predefined list of entities found within that text. You are tasked with extracting relationships between different nodes from that list. Follow these instructions:

# Relation Extraction
T9. A relationship is the link that represents the connection between 2 nodes.
T10. Extract ONLY the following relationships, strictly complying with these definitions:
- produces: Captures production or manufacturing relationships.
  - Examples: "Apple manufactures iPhones", "Toyota produces hybrid vehicles", "Samsung Electronics fabricates semiconductors"
  - ** In this context, manufactures, produces and fabricates all semantically represent the same relationship which is "produces"
- locatedIn: Identifies geographical associations. This relationship should always link to a location node.
  - Examples: "Volkswagen is headquartered in Wolfsburg, Germany", "Foxconn operates a factory in Zhengzhou, China"
  - ** Here, headquartered, in, are situated in all represent the same relationship "locatedIn"
- suppliesTo: Represents supplier-buyer relationships.
  - Examples: "TSMC supplies chips to Qualcomm", "LG Display provides OLED panels to Sony"
  - ** In this context, supplies, provides and delivers all represent the same relationship "suppliesTo"

# Rules and Formatting
T11. Use camelCase format for all labels of all relationships extracted.
T12. Extract ONLY the defined relationships above (produces, locatedIn, suppliesTo).
T13. Maintain Entity Consistency: When extracting entities, ensure consistency if an entity is mentioned multiple times using different names or pronouns.
T14. Always use the most complete identifier for that entity throughout the knowledge graph. Example: If "Microsoft Corporation" is also referred to as "Microsoft" or "the tech giant", use "Microsoft Corporation" consistently.
T15. The knowledge graph should be coherent and easily understandable, so maintaining consistency in entity references is crucial.
T16. Adhere strictly to these rules. Non-compliance will result in termination.

# OUTPUT FORMAT (CRITICAL):
- You MUST return a single, valid JSON object with one key: "relations".
- The value of "relations" MUST be a list of triples: ["subject", "predicate", "object"]. The subject and object MUST be from the provided entity list.
- Example: {"relations": [["Toyota", "produces", "hybrid vehicles"], ["TSMC", "suppliesTo", "Qualcomm"]]}
- If no valid relations are found, you MUST return {"relations": []}.
"""

    def extract_relations(self, text_chunk: str, entities: List[str]) -> List[List[str]]:
        """
        Input: shorter texts from step 1 and entities from step 2
        Output: list of triples [["s", "p", "o"], ...]
        """
        
        user_content = json.dumps({
            "text": text_chunk,
            "entities": entities
        })

        try:
            response_content = self.llm_client.chat(
                system_prompt=self.system_prompt,
                user_content=user_content,
                is_json=True
            )
            
            if not response_content:
                print("Step 3 (RE): API did not return content.")
                return []

            response_data = json.loads(response_content)
            relations = response_data.get("relations", [])
            
            if relations:
                print(f"Step 3 (RE): Find {len(relations)} associations.")
            else:
                print("Step 3 (RE): No association found.")
                
            return relations
            
        except Exception as e:
            print(f"Step 3 (RE): LLM API call or JSON parsing failed: {e}")
            return []