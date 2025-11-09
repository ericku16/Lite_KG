from ..core.clients import BaseLLMClient

class OntologyFilter:
    """
    [Step 1] 使用 ollama 根據 ontology prompt 過濾句子並回傳過濾後的精簡文本 作為step2,3的 input
    Input: entire long texts 
    Output: shorter texts
    """
    
    def __init__(self, llm_client: BaseLLMClient):
        self.llm_client = llm_client
        self.system_prompt = f"""You are an expert in supply chain knowledge extraction. Your task is to analyze the
following text and select only the sentences that are relevant to supply chain entities and relations.

Instructions:
- 1. Keep sentences that mention supply chain–related entities such as: Company, Product, Location.
- 2. Keep sentences that describe supply chain relations, such as: manufactures, produces, fabricates, located_in, operates_in, headquartered_in, supplies_to, provides_to.
- 3. Ignore sentences that only describe unrelated events, finance-only news,politics, or other non-supply chain content.
- 4. Output the selected sentences together as a single paragraph of text (no JSON,no bullet points).

Examples:
[Example 1]
Input Text: "Toyota sources batteries from Panasonic in Japan. The two companies also co-hosted a sports event last year."
Output: "Toyota sources batteries from Panasonic in Japan."

[Example 2]
Input Text: "Apple partners with TSMC to manufacture advanced chips in Taiwan. Tim Cook gave a keynote speech at a university."
Output: "Apple partners with TSMC to manufacture advanced chips in Taiwan."
"""

    def filter_text(self, full_text: str) -> str:
        """
        執行過濾
        
        Input: entire long texts 
        Output: shorter texts
        """
        try:
            filtered_text = self.llm_client.chat(
                system_prompt=self.system_prompt,
                user_content=full_text,
                is_json=False
            )
            
            # 計算並印出過濾效果
            original_len = len(full_text)
            filtered_len = len(filtered_text)
            reduction_percent = (1 - filtered_len / original_len) * 100 if original_len > 0 else 0
            print(f"  ✅ Step 1 (Filter): 句子過濾完成。原文長度: {original_len}, 過濾後長度: {filtered_len} (減少 {reduction_percent:.1f}%)")
            
            return filtered_text
            
        except Exception as e:
            print(f"  ❌ Step 1 (Filter): 句子過濾失敗: {e}. 將使用原始文本")
            return full_text