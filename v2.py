import streamlit as st
from transformers import MBartForConditionalGeneration, MBartTokenizer

# Load the tokenizer and model
model_path = "E:/1Fyafulla/final_model"  # Path to your saved model
tokenizer = MBartTokenizer.from_pretrained(model_path)
model = MBartForConditionalGeneration.from_pretrained(model_path)

# Add custom language code for Tamang if not already present
if 'tmg_TM' not in tokenizer.lang_code_to_id:
    tokenizer.lang_code_to_id['tmg_TM'] = len(tokenizer.lang_code_to_id)

# Language mapping dictionary
language_codes = {
    "English": "en_XX",  # Language code for English
    "Romanized Nepali": "ne_XX",  # Language code for Nepali (Romanized)
    "Nepali": "ne_XX",  # Language code for Nepali
    "Romanized Tamang": "tmg_TM",  # Custom language code for Romanized Tamang
    "Tamang Language": "tmg_TM"  # Custom language code for Tamang
}

# Define the translation function
def translate(input_text, source_lang, target_lang):
    if source_lang not in language_codes or target_lang not in language_codes:
        raise ValueError("Invalid source or target language.")

    # Set source and target languages
    src_lang_code = language_codes[source_lang]
    tgt_lang_code = language_codes[target_lang]
    tokenizer.src_lang = src_lang_code
    tokenizer.tgt_lang = tgt_lang_code

    # Tokenize the input and generate translation
    inputs = tokenizer(input_text, return_tensors="pt", padding=True)
    translated = model.generate(**inputs)

    # Decode the translated text
    translation = tokenizer.decode(translated[0], skip_special_tokens=True)
    
    return translation

# Streamlit App
st.title("Tamang Language Translation")

# Dropdown for source language selection
source_lang = st.selectbox("Select Source Language", ("English", "Romanized Nepali", "Nepali", "Romanized Tamang", "Tamang Language"))

# Dropdown for target language selection
target_lang = st.selectbox("Select Target Language", ("Romanized Tamang", "Tamang Language", "English", "Romanized Nepali", "Nepali"))

# Text input box
input_text = st.text_area("Enter text to translate:")

# Button for translation
if st.button("Translate"):
    if input_text:
        try:
            translation = translate(input_text, source_lang, target_lang)
            st.subheader("Translated Text:")
            st.write(translation)
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.error("Please enter some text to translate.")
