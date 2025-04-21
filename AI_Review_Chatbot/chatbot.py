import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch

# Load RoBERTa model from Hugging Face
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Function to get sentiment using RoBERTa
def get_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    scores = softmax(outputs[0][0].detach().numpy())
    labels = ['Negative', 'Neutral', 'Positive']
    return labels[scores.argmax()]

# Function to generate professional auto-reply based on sentiment
def generate_reply(sentiment):
    if sentiment == "Positive":
        return "✅ Thank you for your kind words. We're delighted to know you're satisfied with our services."
    elif sentiment == "Negative":
        return "⚠️ We sincerely apologize for any inconvenience caused. Our team will review this and take necessary actions."
    else:
        return "ℹ️ Thank you for your honest feedback. We appreciate your input and will continue to improve."

# Streamlit UI
st.set_page_config(page_title="AI Feedback Assistant", layout="centered")

st.title("💬 AI Feedback Assistant for IT Services")
st.markdown("""
Welcome to our **AI-Powered Feedback Assistant**.  
Paste any *client* or *user* feedback related to IT services — our AI will analyze the sentiment and generate a professional, automated response instantly.
""")

# Input box
user_input = st.text_area("📥 Paste feedback or comment from a client, employee, or user:")

# Button to get AI reply
if st.button("Generate AI Response"):
    if user_input.strip():
        sentiment = get_sentiment(user_input)
        reply = generate_reply(sentiment)
        st.write(f"**Detected Sentiment:** `{sentiment}`")
        st.success(f"🤖 Auto-Generated Reply: {reply}")
    else:
        st.warning("Please enter some feedback to analyze.")
