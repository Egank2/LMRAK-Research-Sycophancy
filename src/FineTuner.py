class FineTuner:
    def __init__(self, model, tokenizer, train_dataset, eval_dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

    def fine_tune(self, output_dir="./results", epochs=3, batch_size=16, lr=2e-5):
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
        )
        trainer.train()

# Example Usage
fine_tuner = FineTuner(
    model_handler.model, model_handler.tokenizer,
    train_dataset, dataset_loader.get_split("validation")
)
fine_tuner.fine_tune()