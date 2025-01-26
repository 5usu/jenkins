
import modal
import os

# Define the Modal app
app = modal.App("gpt2-finetune")

# Define the image with updated package versions
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "datasets>=2.15.0",
        "accelerate>=0.25.0",
        "sentencepiece>=0.1.99",
        "pandas>=2.1.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0"
    )
)

# Define a persistent volume
volume = modal.Volume.from_name("datasets-volume")

@app.function(gpu="A100", timeout=3600, volumes={"/data": volume}, image=image)
def finetune():
    from datasets import concatenate_datasets, Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
    import pandas as pd
    import logging

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        # Load GPT-2 model and tokenizer
        logger.info("Loading model and tokenizer...")
        model_name = "gpt2"
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.eos_token_id

        # Load and combine multiple datasets
        dataset_files = [
            "/data/Community Questions Refined.csv",
            "/data/Jenkins Docs QA.csv",
            "/data/QueryResultsUpdated.csv"
        ]

        logger.info("Loading and combining datasets...")
        datasets = []

        # First, let's check the structure of our CSV files
        for file in dataset_files:
            try:
                logger.info(f"Examining file: {file}")
                if not os.path.exists(file):
                    logger.error(f"File not found: {file}")
                    continue

                # Read first few rows to check structure
                df = pd.read_csv(file)
                logger.info(f"Columns in {file}: {df.columns.tolist()}")

                # Convert to datasets format
                dataset = Dataset.from_pandas(df)
                datasets.append(dataset)
                logger.info(f"Successfully converted {file} to Dataset format")
            except Exception as e:
                logger.error(f"Error processing {file}: {str(e)}")
                logger.error("Full error details:", exc_info=True)
                continue

        if not datasets:
            raise ValueError("No datasets were successfully loaded")

        combined_dataset = concatenate_datasets(datasets)
        logger.info(f"Combined dataset columns: {combined_dataset.column_names}")

        # Preprocess dataset
        def preprocess_function(examples):
            # Log available columns
            logger.info(f"Available columns: {list(examples.keys())}")

            # Modify these lines based on your actual column names
            # You might need to adjust these based on the column names in your CSV
            question_col = 'Question' if 'Question' in examples else 'question'  # try both capitalizations
            answer_col = 'Answer' if 'Answer' in examples else 'answer'

            if question_col not in examples or answer_col not in examples:
                raise KeyError(f"Required columns not found. Available columns: {list(examples.keys())}")

            input_texts = examples[question_col]
            output_texts = examples[answer_col]

            # Combine question and answer with a separator
            combined_texts = [f"Question: {q} Answer: {a}" for q, a in zip(input_texts, output_texts)]

            return tokenizer(
                combined_texts,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors="pt"
            )

        logger.info("Tokenizing dataset...")
        tokenized_dataset = combined_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=combined_dataset.column_names
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir="./fine_tuned_model",
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            num_train_epochs=5,
            logging_dir="./logs",
            logging_steps=10,
            save_steps=500,
            save_total_limit=2,
            learning_rate=2e-5,
            weight_decay=0.01,
            fp16=True,
            evaluation_strategy="steps",
            eval_steps=500,
            load_best_model_at_end=True,
        )

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
        )

        # Start fine-tuning
        logger.info("Starting training...")
        trainer.train()

        # Save the fine-tuned model
        logger.info("Saving model...")
        model.save_pretrained("./fine_tuned_model")
        tokenizer.save_pretrained("./fine_tuned_model")

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error("Full error details:", exc_info=True)
        raise

@app.function(volumes={"/data": volume}, image=image)
def list_files():
    print("Contents of /data:")
    for root, dirs, files in os.walk("/data"):
        print(f"\nDirectory: {root}")
        for file in files:
            print(f"  - {file}")

if __name__ == "__main__":
    with app.run():
        # First check the files
        list_files.call()
        # Then run the fine-tuning
        finetune.call()
