import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-3b-4e1t")
model = AutoModelForCausalLM.from_pretrained(
  "stabilityai/stablelm-3b-4e1t"
)

st.title("Text Generation with Hugging Face Transformers")

# Your API token from Hugging Face
api_token = "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"

# Load the tokenizer and model with authentication and trust remote code
model_name = "stabilityai/stablelm-3b-4e1t"
tokenizer = AutoTokenizer.from_pretrained(model_name, revision="main", token=api_token, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
  "stabilityai/stablelm-3b-4e1t",
  trust_remote_code=True,
  torch_dtype="auto",
)
input_prompt = st.text_input("Enter a text prompt:")
generate_button = st.button("Generate Text")

if generate_button:
    if input_prompt:
        inputs = tokenizer(input_prompt, return_tensors="pt")
        tokens = model.generate(
            **inputs,
            max_length=64,
            temperature=0.75,
            top_p=0.95,
            do_sample=True,
        )
        generated_text = tokenizer.decode(tokens[0], skip_special_tokens=True)

        st.write("Generated Text:")
        st.write(generated_text)
    else:
        st.warning("Please enter a text prompt.")

st.info("You need a Hugging Face API token to use this service.")
