from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, TextDataset, \
    DataCollatorForLanguageModeling

# Load pre-trained model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)


# Prepare dataset
def load_dataset(file_path, tokenizer, block_size=128):
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )
    return dataset


# Specify training arguments
training_args = TrainingArguments(
    output_dir="./fine-tuned-distilgpt2_v1",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    ),
    train_dataset=load_dataset("C:/Users/kumar/PycharmProjects/firstIdea/license_data/data.txt", tokenizer)
)

# Train model
trainer.train()

# Save the model and tokenizer
model.save_pretrained("./fine-tuned-gpt2_v1")
tokenizer.save_pretrained("./fine-tuned-gpt2_v1")
