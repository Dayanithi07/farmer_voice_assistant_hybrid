# farmer_voice_assistant_hybrid.py

import streamlit as st
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModelForImageClassification, pipeline
import torch
from gtts import gTTS
import tempfile
import io
import geocoder
import speech_recognition as sr

# -----------------------------
# 1️⃣ Streamlit Config
# -----------------------------
st.set_page_config(
    page_title="பயிர் உதவியாளர்",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
body { background-color: #fdfdfd; color: #000000; }
.stButton>button { background-color: #4CAF50; color: white; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 2️⃣ Load Plant Disease Model
# -----------------------------
model_name = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)
class_labels = {0:"Healthy",1:"Diseased - Class 1",2:"Diseased - Class 2"}  # replace with full labels

# -----------------------------
# 3️⃣ Hugging Face GPT Fallback
# -----------------------------
hf_gpt_pipe = pipeline("text-generation", model="tiiuae/falcon-7b-instruct", device=-1)

# -----------------------------
# 4️⃣ Helper Functions
# -----------------------------
def speak(text):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts = gTTS(text=text, lang='ta')
    tts.save(tmp.name)
    return tmp.name

def is_connected():
    try:
        requests.get("https://www.google.com", timeout=3)
        return True
    except:
        return False

def gpt_response_hybrid(user_text):
    if is_connected():
        try:
            import openai
            openai.api_key = "AIzaSyAU2sKsiu63mPjzMtSH5DhG7_8beUr5F7c"
            response = openai.chat.completions.create(
                model="gemini-1.5-t",
                messages=[
                    {"role": "system", "content": "You are a helpful Tamil-speaking farming assistant."},
                    {"role": "user", "content": user_text}
                ],
                temperature=0.7
            )
            reply_text = response.choices[0].message.content
        except:
            # fallback to Hugging Face GPT if Gemini fails
            output = hf_gpt_pipe(f"Respond in Tamil in a farmer-friendly way:\n{user_text}", max_length=200)
            reply_text = output[0]['generated_text']
    else:
        # offline fallback
        output = hf_gpt_pipe(f"Respond in Tamil in a farmer-friendly way:\n{user_text}", max_length=200)
        reply_text = output[0]['generated_text']

    audio_file = speak(reply_text)
    return reply_text, audio_file

def get_location():
    try:
        g = geocoder.ip('me')
        if g.ok:
            return g.latlng
        return [20.5937, 78.9629]
    except:
        return [20.5937, 78.9629]

def get_weather(lat, lon):
    api_key = "e972e190876a2d130682da01b94f8a80"
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    try:
        response = requests.get(url).json()
        weather_desc = response['weather'][0]['description']
        temp = response['main']['temp']
        return f"{weather_desc}, வெப்பநிலை: {temp}°C"
    except:
        return "வானிலை தரவு கிடைக்கவில்லை"

def get_soil(lat, lon):
    try:
        url = f"https://soilgrids.org/query?lon={lon}&lat={lat}&attributes=bdod,cec,phh2o,ocd"
        response = requests.get(url).json()
        data = response['properties']['layers']
        soil_info = f"BDOD={data['bdod']['values']['mean']}, CEC={data['cec']['values']['mean']}, pH={data['phh2o']['values']['mean']}, Organic Carbon={data['ocd']['values']['mean']}"
        return soil_info
    except:
        return "மண் தகவல் கிடைக்கவில்லை"

def detect_disease(image_file):
    img_bytes = image_file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_idx = logits.argmax(-1).item()
    return class_labels.get(predicted_idx, "Unknown")

def analyze_crop(image_file):
    lat, lon = get_location()
    disease_name = detect_disease(image_file)
    weather_summary = get_weather(lat, lon)
    soil_summary = get_soil(lat, lon)

    detailed_audio_text = f"""
வணக்கம்! உங்கள் பயிர் தற்போதைய நிலையை நான் பார்க்கிறேன்.
பதிவேற்றப்பட்ட புகைப்படத்தின் படி, நோய்: {disease_name}.
உங்கள் பகுதியில் தற்போதைய வானிலை: {weather_summary}.
மண் விவரங்கள்: {soil_summary}.
பயிரை சரியாக பராமரிக்க தேவையான நடவடிக்கைகள்: உங்கள் பயிரின் நோயை கவனித்து மருந்து விதிக்கவும், நீர் அளவை சரிசெய்யவும், செவ்வியல் முறையில் உரம் சேர்க்கவும்.
"""
    audio_file = speak(detailed_audio_text)
    return disease_name, weather_summary, soil_summary, audio_file

def speech_to_text():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("தயவுசெய்து பேசுங்கள்...")
        audio = r.listen(source, phrase_time_limit=10)
    try:
        text = r.recognize_google(audio, language="ta-IN")
        return text
    except:
        return None

# -----------------------------
# 5️⃣ App Layout
# -----------------------------
st.title("🌿 பயிர் உதவியாளர் (Hybrid Voice Assistant)")
st.write("புகைப்படம் பதிவேற்றவும் / கேள்விகள் கேளுங்கள். உங்கள் கேள்விக்கு AI தமிழில் பதில் கூறும்.")

# Voice greeting
if 'greeted' not in st.session_state:
    welcome_audio = speak("வணக்கம்! நான் உங்கள் பயிர் உதவியாளர். புகைப்படம் பதிவேற்றவும் அல்லது கேள்வி கேளுங்கள்.")
    st.audio(welcome_audio)
    st.session_state.greeted = True

# Image upload / camera capture
uploaded_file = st.file_uploader("பயிர் புகைப்படம் பதிவேற்றவும்", type=["jpg","jpeg","png"])
capture_image = st.camera_input("பயிர் புகைப்படம் எடுக்கவும்")
image_to_use = uploaded_file if uploaded_file else capture_image

if image_to_use:
    disease_name, weather_summary, soil_summary, audio_file = analyze_crop(image_to_use)
    
    st.subheader("🔹 நோய் முன்னறிக்கை")
    st.write(disease_name)
    
    st.subheader("🔹 வானிலை")
    st.write(weather_summary)
    
    st.subheader("🔹 மண் தகவல்")
    st.write(soil_summary)
    
    st.audio(audio_file)

# Real-time voice input
st.subheader("🎤 உங்கள் கேள்வி பேசுங்கள்")
if st.button("செயல்படுத்த"):
    spoken_text = speech_to_text()
    if spoken_text:
        st.write(f"நீங்கள் பேசியது: {spoken_text}")
        reply_text, audio_file = gpt_response_hybrid(spoken_text)
        st.write(reply_text)
        st.audio(audio_file)
    else:
        st.error("மிகவும் மன்னிக்கவும், உங்கள் குரல் அறியப்படவில்லை. மறுபடியும் முயற்சிக்கவும்.")
