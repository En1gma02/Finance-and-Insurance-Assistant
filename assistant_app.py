import streamlit as st
import pandas as pd
import os
import requests
import speech_recognition as sr
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Hugging Face API setup
HF_API_KEY_CHAT = os.getenv("HF_API_KEY_CHAT")
HF_API_KEY_TTS = os.getenv("HF_API_KEY_TTS")

CHAT_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
TTS_API_URL = "https://api-inference.huggingface.co/models/microsoft/speecht5_tts"
HEADERS_CHAT = {"Authorization": f"Bearer {HF_API_KEY_CHAT}"}
HEADERS_TTS = {"Authorization": f"Bearer {HF_API_KEY_TTS}"}

# Function to generate AI assistant response
def generate_insurance_assistant_response(prompt_input):
    try:
        payload = {"inputs": prompt_input}
        response = requests.post(CHAT_API_URL, headers=HEADERS_CHAT, json=payload)
        response.raise_for_status()
        result = response.json()
        return result["generated_text"].strip()
    except requests.exceptions.RequestException as e:
        return f"Error: Unable to generate response. {e}"

# Function to transcribe speech input from audio bytes
def transcribe_speech(audio_bytes):
    try:
        r = sr.Recognizer()
        with sr.AudioFile(BytesIO(audio_bytes)) as source:
            audio_data = r.record(source)
        text = r.recognize_google(audio_data)
        return text
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        return None

# Define the AI Assistant page
def ai_assistant_page():
    st.title('AI Assistant')
    st.write("Your personal insurance and finance expert")

    # Initialize session state for messages
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            container_class = "user-container" if message['role'] == "user" else "assistant-container"
            st.markdown(f"""
            <div class="{container_class}">
                <p><strong>{'You' if message['role'] == 'user' else 'Assistant'}:</strong> {message['content']}</p>
            </div>
            """, unsafe_allow_html=True)

    # Handle user input and generate responses
    user_input = st.text_input("Type your message here:")
    if st.button("Send") and user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        response = generate_insurance_assistant_response(user_input)
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Handle speech input
    st.write("Or speak to your assistant:")
    audio_bytes = st.audio_recorder(text="Start recording", format="audio/wav")
    if audio_bytes:
        speech_text = transcribe_speech(audio_bytes)
        if speech_text:
            st.session_state.messages.append({"role": "user", "content": speech_text})
            response = generate_insurance_assistant_response(speech_text)
            st.session_state.messages.append({"role": "assistant", "content": response})

# Run the app
if __name__ == "__main__":
    ai_assistant_page()
