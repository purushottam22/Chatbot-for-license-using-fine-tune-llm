from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the fine-tuned model and tokenizer
model_directory = "C:/Users/kumar/PycharmProjects/firstIdea/model/kaggle/distil_gpt_29_07"

model = GPT2LMHeadModel.from_pretrained(model_directory)
tokenizer = GPT2Tokenizer.from_pretrained(model_directory)

# Now you can use the model for inference or further fine-tuning
input_text = "what is apache license ?"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Generate text
output = model.generate(input_ids, max_length=100)
# print(output)
print("answer : ", tokenizer.decode(output[0], skip_special_tokens=True))
