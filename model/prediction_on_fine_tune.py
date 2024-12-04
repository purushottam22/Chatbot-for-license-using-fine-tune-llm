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
    # print("answer : ", response)
    return response


model, tokenize = load_model()

import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="License Chatbot",
    page_icon="🤖",
    layout="centered",
)

# Apply background image using CSS
page_bg_color = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #F5F5F5; /* Light gray color */
    color: #333333; /* Text color */
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
[data-testid="stSidebar"] {
    background: rgba(0,0,0,0);
}
</style>
"""

st.markdown(page_bg_color, unsafe_allow_html=True)

# Page title
st.title("License Chatbot")


user_query = st.text_input("Enter your question about licenses:", "")

# Button to submit query
if st.button("Submit"):
    if user_query.strip():
        # Dummy response (replace this with your chatbot response function)
        response = quet_ans(user_query, model, tokenize)
        # response = f"Answer for your question: '{response}'"
        st.text_area("Chatbot Response:", value=response, height=100)
    else:
        st.warning("Please enter a question.")

