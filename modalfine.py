import modal
from datasets import load_dataset, concatenate_datasets

# Define the Modal image with necessary dependencies
image = modal.Image.debian_slim().pip_install(
    "transformers",
    "datasets",
    "accelerate",
    "torch",
    "sentencepiece"
)

# Define the Modal app
app = modal.App("gpt2-finetune", image=image)

# Define a persistent volume
volume = modal.Volume.from_name("datasets-volume")

# Define the function to run the fine-tuning
@app.function(gpu="A100", timeout=3600, volumes={"/data": volume})
def finetune():
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

    # Load GPT-2 model and tokenizer
    model_name = "gpt2"  # Use the public GPT-2 model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load and combine multiple datasets
    dataset_files = [
        "/data/Community Questions Refined.csv",
        "/data/Jenkins Docs QA.csv",
        "/data/QueryResultsUpdated.csv"
    ]
    datasets = [load_dataset('csv', data_files=file)['train'] for file in dataset_files]
    combined_dataset = concatenate_datasets(datasets)

    # Preprocess dataset
    def preprocess_function(examples):
        return tokenizer(examples['instruction'], examples['response'], truncation=True, padding='max_length')

    tokenized_dataset = combined_dataset.map(preprocess_function, batched=True)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./fine_tuned_model",
        per_device_train_batch_size=4,
        num_train_epochs=5,
        logging_dir="./logs",
        save_steps=500,
        save_total_limit=2,
        fp16=True,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    # Start fine-tuning
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")

# Entrypoint for running the function
if __name__ == "__main__":
    with app.run():
        finetune.call()
