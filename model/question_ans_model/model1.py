import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset

# Define your dataset class
class QADataset:
    def __init__(self, data_file, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        answer = item['answer']

        inputs = self.tokenizer(
            question,
            answer,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze()
        }


# Path to your JSON file containing questions and answers
data_file = 'C:/Users/kumar/PycharmProjects/firstIdea/license_data/quest_ans.json'

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Add padding token to tokenizer
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))

# Create dataset
dataset = QADataset(data_file, tokenizer)

# Convert custom dataset to Hugging Face Dataset
dataset = Dataset.from_dict({
    'input_ids': [dataset[i]['input_ids'] for i in range(len(dataset))],
    'attention_mask': [dataset[i]['attention_mask'] for i in range(len(dataset))]
})

# Data collator for dynamic padding
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Training arguments
training_args = TrainingArguments(
    output_dir='./fine-tuned-gpt2',
    overwrite_output_dir=True,
    num_train_epochs=50,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=100,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained('./fine-tuned-gpt2')
tokenizer.save_pretrained('./fine-tuned-gpt2')
