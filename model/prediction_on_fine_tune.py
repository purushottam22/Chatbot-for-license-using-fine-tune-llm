import warnings

from transformers import GPT2LMHeadModel, GPT2Tokenizer

warnings.filterwarnings("ignore", category=UserWarning)


def load_model():
    # Load the fine-tuned model and tokenizer
    model_directory = "C:/Users/kumar/PycharmProjects/firstIdea/model/kaggle/final_distilgpt_100"

    model = GPT2LMHeadModel.from_pretrained(model_directory)
    tokenizer = GPT2Tokenizer.from_pretrained(model_directory)
    return model, tokenizer


# print("here")
# Now you can use the model for inference or further fine-tuning

def quet_ans(input_text, model, tokenizer):
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # Generate text
    output = model.generate(
        input_ids,
        max_length=1000,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    response = response[len(input_text):].strip()
    print("answer : ", response)


model, tokenize = load_model()

while True:
    input_text = input("User : ")
    quet_ans(input_text, model, tokenize)
