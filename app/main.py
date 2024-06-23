import streamlit as st
from llm_utils import OpenAIRetrieval, CONTEXT_TEMPLATE, PROMPT_TEMPLATE
import time

def format_prompt(prompt):
    return f"Response for the prompt: {prompt}"

def main():
    st.title("DTC Q&A System")
    oar = OpenAIRetrieval()

    with st.form(key='rag_form'):
        prompt = st.text_input("Enter your prompt")
        response_placeholder = st.empty()
        submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        response_placeholder.markdown("Loading...")
        time.sleep(2)
        response = oar.qa_bot(prompt, CONTEXT_TEMPLATE, PROMPT_TEMPLATE)
        formated_response = format_prompt(response)
        response_placeholder.markdown(formated_response)


if __name__ == "__main__":
    main()
