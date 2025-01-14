class Evaluator:
    def __init__(self, trainer):
        self.trainer = trainer

    def evaluate(self):
        results = self.trainer.evaluate()
        print(f"Evaluation Results: {results}")

# Example Usage
evaluator = Evaluator(fine_tuner)
evaluator.evaluate()