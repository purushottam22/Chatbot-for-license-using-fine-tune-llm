![Chatbot UI](https://github.com/purushottam22/Chatbot-for-license-using-fine-tune-llm/blob/dev/UI.png)

# License Chatbot  

**License Chatbot** is a conversational AI designed to assist developers and users in understanding various software licenses. Software licensing is a critical aspect of software development, and a lack of knowledge can lead to potential copyright and compliance issues. This chatbot provides answers to questions related to different licenses used in software development.

## Supported Licenses  
The chatbot provides information about the following licenses:  
- **MIT License**  
- **Apache License**  
- **ISC License**  
- **GNU General Public License (GPL)**  
- **Mozilla Public License (MPL)**  
- **zlib License**  
- **BSD License**  
- **Server Side Public License (SSPL)**  
- **GNU Affero General Public License (AGPL)**  
- **GNU Lesser General Public License (LGPL)**  
- **Eclipse Public License (EPL)**  
- **Common Development and Distribution License (CDDL)**  
- **Artistic License**  
- **The Unlicense**  
- **Boost Software License**  
- **University of Illinois/NCSA Open Source License (NCSA License)**  
- **Common Public License (CPL)**  
- **Microsoft Public License**  
- **Open Software License (OSL)**  

## Methodology  
1. **Data Collection**:  
   We have curated a dataset consisting of frequently asked questions and answers related to the supported licenses.  

2. **Model Fine-Tuning**:  
   A language model (LLM) was fine-tuned using the collected dataset to improve its performance in answering license-related queries accurately.  

3. **User Interface**:  
   A web-based interface is built using **Streamlit** to provide an easy-to-use platform for interacting with the chatbot.

---

## Installation  

To set up the chatbot, follow these steps:  

1. Clone the repository:  
   ```bash
   git clone <[repository_link](https://github.com/purushottam22/Chatbot-for-license-using-fine-tune-llm/tree/main)>
   cd <Chatbot-for-license-using-fine-tune-llm>
   ```

2. Install the required Python modules:  
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Chatbot  

1. Navigate to the `Model` folder:  
   ```bash
   cd Model
   ```

2. Start the chatbot UI using Streamlit:  
   ```bash
   streamlit run prediction_on_fine_tune.py
   ```

3. Open the link provided in the terminal to access the chatbot interface in your web browser.

---

## Features  
- **Interactive Q&A**: Get instant answers to your questions about different software licenses.  
- **Comprehensive Coverage**: Covers a wide range of open-source and proprietary licenses.  
- **User-Friendly Interface**: Simple and intuitive interface for easy interaction.  

---

## Future Improvements  
- Addition of more licenses and related questions.  
- Integration with additional language models for better performance.  
- Support for multi-language queries.  

---

## Contributions  
Contributions are welcome! Feel free to submit a pull request or open an issue if you have suggestions for improvement.

---

By using this chatbot, developers can ensure they understand the licensing implications of the software they use, avoiding legal risks and fostering compliance with open-source and proprietary licensing terms.
