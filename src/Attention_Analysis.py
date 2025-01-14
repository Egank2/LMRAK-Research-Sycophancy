class AttentionAnalysis:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def visualize_attention(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").to(device)
        outputs = self.model(**inputs, output_attentions=True)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        head_view(outputs.attentions, tokens)

# Example Usage
attention_analysis = AttentionAnalysis(model_handler.model, model_handler.tokenizer)
attention_analysis.visualize_attention(test_text)