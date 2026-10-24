import streamlit as st
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

st.set_page_config(page_title="WildGuard AI", page_icon="🐾", layout="centered")

st.title("🐾 WildGuard AI")
st.write("Discover wildlife with AI. Identify five supported animals or flag images that appear to show something else.")

SUPPORTED = {
    "cat": ["tabby", "persian", "siamese", "egyptian cat", "cat"],
    "elephant": ["elephant", "tusker"],
    "lion": ["lion", "lioness"],
    "panda": ["panda"],
    "tiger": ["tiger"]
}

@st.cache_resource
def load_model():
    proc = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    mod = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")
    return proc, mod

processor, model = load_model()

uploaded = st.file_uploader("Choose a clear image", type=["jpg","png","webp","jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, width=350)
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    idx = logits.argmax(-1).item()
    label = model.config.id2label[idx].lower()
    conf = torch.softmax(logits, dim=1)[0, idx].item()*100

    found = None
    for animal, keys in SUPPORTED.items():
        if any(k in label for k in keys):
            found = animal
            break
    
    if found:
        st.success(f"✅ Identified: {found.upper()} ({conf:.1f}%) - Model: {label}")
    else:
        st.error(f"🚩 Flagged: This is {label} - Not in 5 supported animals ({conf:.1f}%)")
