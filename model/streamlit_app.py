from transformers import GPT2LMHeadModel, GPT2Tokenizer
import streamlit as st


def load_model(model_dir):
    try:
        # Load the fine-tuned model and tokenizer
        model = GPT2LMHeadModel.from_pretrained(model_dir)
        tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


def get_response(input_text, tokenizer, model):
    try:
        input_ids = tokenizer.encode(input_text, return_tensors='pt')
        output = model.generate(input_ids, max_length=100)
        return tokenizer.decode(output[0], skip_special_tokens=True)
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return ""


def main():
    model, tokenizer = load_model("C:/Users/kumar/PycharmProjects/firstIdea/model/fine-tuned-gpt2_v1")

    st.title("License Question Answer")

    # Initialize session state for conversation history
    if "history" not in st.session_state:
        st.session_state.history = []

    # Display conversation history in a scrollable container
    st.subheader("Conversation History")
    chat_container = st.container()
    with chat_container:
        for entry in st.session_state.history:
            st.markdown(f"**You**: {entry['user']}", unsafe_allow_html=True)
            st.markdown(f"**Bot**: {entry['bot']}", unsafe_allow_html=True)
            st.write("---")

    # Button to clear chat history
    if st.button("Clear Chat"):
        st.session_state.history = []
        st.experimental_rerun()

    # User input
    st.subheader("Your Input")
    user_input = st.text_input("You:", key='input_box')

    if st.button("Send"):
        if user_input:
            # Get chatbot response
            response = get_response(user_input, tokenizer, model)
            # print(response)

            # Update conversation history
            st.session_state.history.append({"user": user_input, "bot": response})

            # Clear the input box after submission
            # st.session_state["input_box"] = ""

            # Rerun the app to display the updated conversation
            st.rerun()


if __name__ == "__main__":
    main()

# Now you can use the model for inference or further fine-tuning
# input_text = "what is apache license ?"
