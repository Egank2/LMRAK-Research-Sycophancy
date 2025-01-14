class ModelHandler:
    def __init__(self, model_name="gpt-2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)

    def forward_pass(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").to(device)
        outputs = self.model(**inputs)
        return outputs

# Initialize the model
model_handler = ModelHandler(model_name="gpt-2")