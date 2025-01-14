class TaskVector:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def extract_vector(self, text1, text2):
        def get_activations(text):
            inputs = self.tokenizer(text, return_tensors="pt").to(device)
            outputs = self.model(**inputs, output_hidden_states=True)
            return outputs.hidden_states[-1].detach()

        activation1 = get_activations(text1)
        activation2 = get_activations(text2)
        return activation2 - activation1

# Example Usage
task_vector = TaskVector(model_handler.model, model_handler.tokenizer)
vector = task_vector.extract_vector("I agree with you.", "Evidence shows this is incorrect.")