import streamlit as st
import pandas as pd
import os
import io
from dotenv import load_dotenv
import speech_recognition as sr
from audio_recorder_streamlit import audio_recorder
import requests
from io import BytesIO

# Load environment variables from .env file
load_dotenv()

# Hugging Face API setup
HF_API_KEY_CHAT = "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"
HF_API_KEY_TTS = "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"

CHAT_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
TTS_API_URL = "https://api-inference.huggingface.co/models/microsoft/speecht5_tts"
HEADERS_CHAT = {"Authorization": f"Bearer {HF_API_KEY_CHAT}"}
HEADERS_TTS = {"Authorization": f"Bearer {HF_API_KEY_TTS}"}

# Function to load fine-tuning data and bot_score.csv
def load_data():
    fine_tuning_file_path = 'D:/bot/tune_data.txt'
    csv_file_path = 'D:/bot/bot_score.csv'

    fine_tuning_data = ""
    fitness_discount_data = {}

    if os.path.exists(fine_tuning_file_path):
        with open(fine_tuning_file_path, 'r') as file:
            fine_tuning_data = file.read()

    if os.path.exists(csv_file_path):
        df = pd.read_csv(csv_file_path)
        for index, row in df.iterrows():
            fitness_score = row['Fitness Score']
            discount = row['Discount']
            fitness_discount_data[fitness_score] = discount

    return fine_tuning_data, fitness_discount_data

# Function to clear chat history
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Function to predict discount based on fitness score
def predict_discount(fitness_score):
    if fitness_score >= 90:
        return 30  # 30% discount
    elif fitness_score >= 80:
        return 25  # 25% discount
    elif fitness_score >= 70:
        return 20  # 20% discount
    elif fitness_score >= 60:
        return 15  # 15% discount
    elif fitness_score >= 50:
        return 10  # 10% discount
    elif fitness_score >= 40:
        return 5  # 5% discount
    else:
        return 0  # No discount

# Function to generate AI assistant response
def generate_insurance_assistant_response(prompt_input, fine_tuning_data, fitness_discount_data):
    system_message = "You are a consultant with expertise in personal finance and insurance. Provide crisp and short responses."

    if fine_tuning_data:
        system_message += f"\n\nFine-tuning data:\n{fine_tuning_data}"

    if "fitness score" in prompt_input.lower() or "discount" in prompt_input.lower():
        return "Please provide your fitness score to get information about the discount you qualify for."

    try:
        user_fitness_score = float(prompt_input)
        discount = predict_discount(user_fitness_score)
        return f"Your fitness score is {user_fitness_score}. Based on this, you get {discount}% discount."
    except ValueError:
        pass

    try:
        messages = [{"role": "system", "content": system_message}, {"role": "user", "content": prompt_input}]
        payload = {"inputs": {"past_user_inputs": [], "generated_responses": [], "text": prompt_input}}
        response = requests.post(CHAT_API_URL, headers=HEADERS_CHAT, json=payload)
        response.raise_for_status()
        result = response.json()
        return result["generated_text"].strip()
    except Exception as e:
        return f"Error: Unable to generate response. {e}"

def text_to_speech(text):
    payload = {"inputs": text}
    response = requests.post(TTS_API_URL, headers=HEADERS_TTS, json=payload)
    return response.content if response.status_code == 200 else None

def transcribe_speech(audio_bytes):
    try:
        r = sr.Recognizer()
        audio_file = sr.AudioFile(io.BytesIO(audio_bytes))
        with audio_file as source:
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

    # Custom CSS for chat containers
    st.markdown("""
    <style>
    .user-container {
        background-color: #2b5c8a;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
    }
    .assistant-container {
        background-color: #1e3d5a;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
    }
    .chat-text {
        color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

    # Load data and configurations
    fine_tuning_data, fitness_discount_data = load_data()

    # Define sidebar for AI Assistant configurations
    with st.sidebar:
        st.title('🏛️🔍 AI-Assistant Settings')
        st.button('Clear Chat History', on_click=clear_chat_history)

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
                <p class="chat-text"><strong>{'You' if message['role'] == 'user' else 'Assistant'}:</strong> {message['content']}</p>
            </div>
            """, unsafe_allow_html=True)

    # Handle user input and generate responses
    user_input_container = st.container()
    with user_input_container:
        user_input_col, mic_col = st.columns([0.8, 0.2])
        with user_input_col:
            user_input = st.text_input("Type your message here:", key="user_input")
        with mic_col:
            audio_bytes = audio_recorder(text="", icon_size="2x")
            if audio_bytes:
                speech_text = transcribe_speech(audio_bytes)
                if speech_text:
                    st.session_state.user_input = speech_text
                    st.experimental_rerun()

        if st.button("Send"):
            st.session_state.messages.append({"role": "user", "content": user_input})
            response = generate_insurance_assistant_response(user_input, fine_tuning_data, fitness_discount_data)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.experimental_rerun()

    # Handle speech input
    if "user_input" in st.session_state:
        st.session_state.messages.append({"role": "user", "content": st.session_state.user_input})
        response = generate_insurance_assistant_response(st.session_state.user_input, fine_tuning_data, fitness_discount_data)
        st.session_state.messages.append({"role": "assistant", "content": response})
        del st.session_state.user_input
        st.experimental_rerun()

    # Add a single button for text-to-speech for the most recent assistant response
    if st.session_state.messages and st.session_state.messages[-1]['role'] == "assistant":
        if st.button("🔊 Listen to Last Response"):
            last_response = st.session_state.messages[-1]['content']
            audio_bytes = text_to_speech(last_response)
            if audio_bytes:
                st.audio(audio_bytes, format='audio/wav')

# Run the app
if __name__ == "__main__":
    ai_assistant_page()
