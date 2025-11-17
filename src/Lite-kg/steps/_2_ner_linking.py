import time
import requests
from flair.data import Sentence
from flair.models import SequenceTagger
from typing import List, Dict, Any
from ..core.models import NERLinkResult

class NERLinker:
    """
    [Step 2 & 2.5] Execute Flair NER and use the Wikidata API to link entities.

    """

    def __init__(self, ner_model_path: str):
        self.tagger = self._load_ner_model(ner_model_path)
        self.wikidata_headers = {'User-Agent': 'Lite-KG/1.0 (your_email@example.com)'}
        self.wikidata_api_endpoint = "https://www.wikidata.org/w/api.php"
        self.total_linking_time = 0.0

    def _load_ner_model(self, model_path: str):
        """Load Flair NER model"""
        try:
            tagger = SequenceTagger.load(model_path)
            print(f" The Flair NER model was successfully loaded from '{model_path}'")
            return tagger
        except Exception as e:
            print(f" Failed to load Flair NER model:{e}")
            return None

    def _get_ner_entities(self, text: str) -> List[str]:
        """
        [Step 2] Extract entities from the text after the ontology filter using the Flair NER model.
        Input: shorter texts from step 1 
        Output: entities
        """
        if not self.tagger:
            print("Step 2 (NER): NER model not loaded")
            return []
            
        sentence = Sentence(text)
        self.tagger.predict(sentence)
        entities = sentence.get_spans('ner')
        
        raw_entity_texts = [entity.text for entity in entities]
        if len(raw_entity_texts) < 2:
            print(f"Step 2 (NER): Find fewer than 2 entities")
            return []
            
        print(f"Step 2 (NER): Find {len(raw_entity_texts)} raw entities: {raw_entity_texts[:5]}...")
        return raw_entity_texts

    def _get_wikidata_info_from_api(self, entity_name: str) -> Dict[str, Any]:
        """
        [Step 2.5] Using the Wikidata API and design candidate sorting
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
                print(f"The API failed to find any candidates for '{entity_name}'.")
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
                print(f"The API found the best match for'{entity_name}' (score: {highest_score}): {q_id} - {final_label} ({desc})")
                return {
                    'name': entity_name,
                    "canonical_name": final_label,
                    "wikidata_id": q_id
                }
            else:
                print(f"No candidate for'{entity_name}'was found in the API.")
                return None

        except requests.exceptions.RequestException as e:
            print(f"A network error occurred while calling the Wikidata API:{e}")
            return None

    def link_entities(self, text: str) -> NERLinkResult:
        """
        [Step 2 & 2.5] Perform NER and make entity connections
        """
        # Step 2: NER
        raw_entity_texts = self._get_ner_entities(text)
        if not raw_entity_texts:
            return NERLinkResult(lookup_map={}, canonical_entities_for_re=[])

        # Step 2.5: Entity Linking
        print("Step 2.5 (Linking): Entity linking is being performed using the Wikidata API.")
        linking_start = time.time()
        
        entity_map = {}
        unique_entities = sorted(list(set(raw_entity_texts)), key=str.lower)

        for entity_name in unique_entities:
            api_result = self._get_wikidata_info_from_api(entity_name)
            
            if api_result:
                entity_map[entity_name] = api_result
            else:
                # Retain the original name as the standard name even if the link fails
                # Maintain structural consistency, including the name key
                entity_map[entity_name] = {
                    "name": entity_name,
                    "canonical_name": entity_name, 
                    "wikidata_id": None
                }
            
            time.sleep(0.1) 

        linking_end = time.time()
        linking_duration = linking_end - linking_start
        self.total_linking_time += linking_duration
        print(f"Step 2.5 (Linking): This link took: {linking_duration:.2f} seconds.")

        # Create a lookup map and standardized entity list for the RE step.
        lookup = {}
        for raw_name, info in entity_map.items():
            if info and info.get('canonical_name'):
                lookup[raw_name.lower()] = info
                lookup[info['canonical_name'].lower()] = info
        
        # Extract the canonical_name directly from the entity_map to avoid duplicate processing.
        canonical_entities_for_re = sorted(
            list(set(
                info['canonical_name'] 
                for info in entity_map.values() 
                if info and info.get('canonical_name')
            )),
            key=str.lower
        )
        
        print(f"Step 2.5 (Linking): Standardized entities ({len(canonical_entities_for_re)}: {canonical_entities_for_re[:5]}...")

        return NERLinkResult(
            lookup_map=lookup,
            canonical_entities_for_re=canonical_entities_for_re
        )