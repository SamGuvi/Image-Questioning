from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image
import streamlit as st

# prepare image + question
url = st.text_input("Enter the Image URL")
if url is not None: 
    image = Image.open(requests.get(url, stream=True).raw)
    st.image(image, width=500)


text = st.text_input("Enter the Question")

bt=st.button("Predict")
if bt:
    
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

    # prepare inputs
    encoding = processor(image, text, return_tensors="pt")

    # forward pass
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    st.success(f"Predicted answer: {model.config.id2label[idx]}" )
else:
    st.warning("Please select an option")