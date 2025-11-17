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
    The main knowledge graph extractor
    This category initializes and coordinates all steps.
    """

    def __init__(self, 
        provider: str,
        model_name: str, 
        ner_model_path: str, 
        api_key: str = None):

        """
        Args:
            provider (str): openai or ollama
            model_name (str): Model name (e.g., "gpt-4o", "mistral:latest")
            ner_model_path (str): Path to the local Flair NER model
            api_key (str, optional): OpenAI API
        """
        print(f"--- Initializing Extractor ---")
        print(f"  Provider: {provider}")
        print(f"  LLM Model: {model_name}")
        print(f"  NER Model: {ner_model_path}")
        
        # 1. Initialize the LLM client (for Step 1 and Step 3)
        self.llm_client = get_llm_client(provider, model_name, api_key)
        
        # 2. Initialize all steps
        self.filterer = OntologyFilter(self.llm_client)
        self.ner_linker = NERLinker(ner_model_path)
        self.extractor = RelationExtractor(self.llm_client)
        print("--- Initialization complete ---\n")

    def process_documents(
        self, 
        folder_path: str, 
        output_json_path: str, 
        chunk_size: int = 5000, 
        delay_between_chunks: int = 1
    ) -> List[Triple]:
        """
        Process all files in the folder and extract relation triples
        """
        
        print(f"\n--- Start processing the folder '{folder_path}' ---")
        output_dir = os.path.dirname(output_json_path)
        if not os.path.exists(output_dir): 
            os.makedirs(output_dir)
            
        all_extracted_triples: List[Triple] = []
        start_time = time.time()

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if not (os.path.isfile(file_path) and (filename.endswith(".txt") or filename.endswith(".pdf"))):
                continue
                
            print(f"\n Files being processed: {filename}")
            
            # Use utility functions to read files.
            content = load_document(file_path)
            if not content:
                print(f"The file content is empty. Skip {filename}")
                continue

            try:
                # Step 1: Ontology Filtering
                filtered_content = self.filterer.filter_text(content)
                if not filtered_content or not filtered_content.strip():
                    print("No relevant content found after filtering.")
                    continue

                # 切分文本
                text_chunks = [filtered_content[i:i + chunk_size] for i in range(0, len(filtered_content), chunk_size)]
                
                for i, chunk in enumerate(text_chunks):
                    print(f"\n Block being processed {i+1}/{len(text_chunks)}...")

                    # Step 2 & 2.5: NER + Entity Linking
                    # ner_link_result Includes .lookup_map and .canonical_entities_for_re
                    ner_link_result = self.ner_linker.link_entities(chunk)
                    
                    if not ner_link_result.canonical_entities_for_re:
                        print("  - Step 2.5 (Linking): Not enough entities found.")
                        continue

                    # Step 3: LLM-RE 
                    # Relationship extraction is performed using a standardized entity list.
                    relations = self.extractor.extract_relations(
                        chunk, 
                        ner_link_result.canonical_entities_for_re
                    )
                    
                    if relations:
                        print(f"Final formatting in progress")
                        formatted_count = 0
                        for rel in relations:
                            if isinstance(rel, list) and len(rel) == 3:
                                s, p, o = rel
                                
                                # Find the Wikidata ID from the lookup map based on the standardized name returned by the RE.
                                s_info = ner_link_result.lookup_map.get(s.lower())
                                o_info = ner_link_result.lookup_map.get(o.lower())
                                
                                s_canonical = s_info['canonical_name'] if s_info else s
                                s_id = s_info['wikidata_id'] if s_info else None
                                
                                o_canonical = o_info['canonical_name'] if o_info else o
                                o_id = o_info['wikidata_id'] if o_info else None
                                
                                # Using the Pydantic model
                                final_triple = Triple(
                                    subject=EntityDetail(name=s, canonical_name=s_canonical, wikidata_id=s_id),
                                    predicate=p,
                                    object=EntityDetail(name=o, canonical_name=o_canonical, wikidata_id=o_id)
                                )
                                all_extracted_triples.append(final_triple)
                                formatted_count += 1

                        print(f"Successfully processed {formatted_count} associations and added them to the Wikidata ID.")

                    if i < len(text_chunks) - 1:
                        time.sleep(delay_between_chunks)
                        
            except Exception as e:
                print(f"A critical error occurred while processing file {filename}:{e}")

        print(f"\n--- Writing the results to '{output_json_path}'... ---")
        
        # Convert the Pydantic model into a list of dictionaries for JSON storage.
        results_data = [triple.model_dump() for triple in all_extracted_triples]
        
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"JSON file successfully generated. A total of {len(all_extracted_triples)} related entries were extracted.")

        # Display total time 
        end_time = time.time()
        elapsed_time = end_time - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print("\n" + "="*50)
        print(f"Total time: {minutes} m {seconds} s ({elapsed_time:.2f} s)")
        
        # Displaying the total time taken to connect entities
        linking_minutes = int(self.ner_linker.total_linking_time // 60)
        linking_seconds = int(self.ner_linker.total_linking_time % 60)
        print(f"Total time spent on entity link (API query): {linking_minutes} m {linking_seconds} s ({self.ner_linker.total_linking_time:.2f} s)")
        print("="*50)

        return all_extracted_triples