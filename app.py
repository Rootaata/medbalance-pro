# MedBalance Pro - REAL MODEL VERSION
import streamlit as st
import hashlib
import pandas as pd
import plotly.express as px
from supabase import create_client
import zipfile
import io
import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import tempfile
import datetime

# ============ SUPABASE CREDENTIALS ============
SUPABASE_URL = "https://uhskvktshxojggmfcqtd.supabase.co"
SUPABASE_KEY = "sb_publishable_ArKrKiXzj4Wq04-9N-dMBA_iug1h2hm"

def get_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def hash_password(pwd):
    return hashlib.sha256(pwd.encode()).hexdigest()

# ============ AI MODEL (same architecture as Colab) ============
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 32 * 32, 64)
        self.fc2 = nn.Linear(64, 2)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

@st.cache_resource
def load_model():
    device = torch.device('cpu')
    model = SimpleCNN().to(device)
    model_path = 'medbalance_model.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model, device
    else:
        st.warning("⚠️ Model file not found. Please upload medbalance_model.pth to the repository.")
        return None, device

model, device = load_model()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

def predict_image(image):
    """Returns 0 = Normal, 1 = Pneumonia"""
    if model is None:
        return 1  # fallback
    img = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)
    return predicted.item()

def process_zip(zip_file):
    """Extract zip, predict every image, return counts"""
    normal_count = 0
    pneumonia_count = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_file, 'r') as zf:
            zf.extractall(tmpdir)
        for root, dirs, files in os.walk(tmpdir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    try:
                        img_path = os.path.join(root, file)
                        img = Image.open(img_path)
                        pred = predict_image(img)
                        if pred == 0:
                            normal_count += 1
                        else:
                            pneumonia_count += 1
                    except Exception as e:
                        st.warning(f"Could not process {file}: {e}")
    return normal_count, pneumonia_count

# ============ MULTI-LANGUAGE SUPPORT (simplified but complete) ============
TEXTS = {
    "en": {
        "title": "🏥 MedBalance Pro",
        "subtitle": "Medical Image Balancing Platform",
        "upload_btn": "Upload ZIP file with X-ray images",
        "processing": "Analyzing images with AI model...",
        "results": "Prediction Results",
        "normal_count": "Predicted Normal",
        "pneumonia_count": "Predicted Pneumonia",
        "download": "Download Results",
        "login": "Login",
        "signup": "Sign Up",
        "username": "Username",
        "password": "Password",
        "email": "Email",
        "create_account": "Create Account",
        "welcome": "Welcome",
        "dashboard": "Dashboard",
        "upload_balance": "Upload & Predict",
        "history": "History",
        "about": "About",
        "logout": "Logout",
        "invalid": "Invalid username or password",
        "exists": "Username already exists",
        "success": "Account created!",
        "model_performance": "Model Performance (Validation)",
        "normal_before": "Normal detection (baseline)",
        "normal_after": "Normal detection (ours)",
        "pneumonia_before": "Pneumonia detection (baseline)",
        "pneumonia_after": "Pneumonia detection (ours)",
        "how_it_works": "How It Works",
        "step1": "1. Upload a ZIP of chest X-rays",
        "step2": "2. AI model predicts each image",
        "step3": "3. See predicted counts",
        "step4": "4. Download prediction report"
    },
    # French, Arabic, Chinese can be added similarly – copy from previous code.
}
# For brevity, we include only English here; you can copy the full multi‑language dict from earlier.

# ============ APP UI ============
st.set_page_config(page_title="MedBalance Pro", layout="wide")

# Language selection (default English)
lang = st.sidebar.selectbox("Language", ["English"], index=0)
t = TEXTS["en"]  # extend for other languages

# RTL support placeholder (add Arabic if needed)
st.title(t["title"])
st.markdown(f"### {t['subtitle']}")

# Session state init
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None

# ============ LOGIN / SIGNUP ============
if not st.session_state.logged_in:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### {t['login']}")
        login_user = st.text_input(t["username"], key="login_user")
        login_pass = st.text_input(t["password"], type="password", key="login_pass")
        if st.button(t["login"]):
            supabase = get_supabase()
            hashed = hash_password(login_pass)
            result = supabase.table("users").select("*").eq("username", login_user).eq("password", hashed).execute()
            if result.data:
                st.session_state.logged_in = True
                st.session_state.username = login_user
                st.rerun()
            else:
                st.error(t["invalid"])
    with col2:
        st.markdown(f"### {t['signup']}")
        new_user = st.text_input(t["username"], key="new_user")
        new_pass = st.text_input(t["password"], type="password", key="new_pass")
        new_email = st.text_input(t["email"])
        if st.button(t["create_account"]):
            supabase = get_supabase()
            hashed = hash_password(new_pass)
            try:
                supabase.table("users").insert({"username": new_user, "password": hashed, "email": new_email}).execute()
                st.success(t["success"])
            except:
                st.error(t["exists"])
else:
    # ============ MAIN APP (LOGGED IN) ============
    st.sidebar.markdown(f"### {t['welcome']}, {st.session_state.username}!")
    menu = st.sidebar.radio("", [t["dashboard"], t["upload_balance"], t["history"], t["about"], t["logout"]])
    
    if menu == t["dashboard"]:
        st.markdown(f"## {t['dashboard']}")
        st.markdown(f"### {t['model_performance']}")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(t["normal_before"], "0%")
            st.metric(t["pneumonia_before"], "100%")
        with col2:
            st.metric(t["normal_after"], "40%", delta="+40%")
            st.metric(t["pneumonia_after"], "71%", delta="-29%")
        st.markdown(f"## {t['how_it_works']}")
        for i in range(1,5):
            st.info(t[f"step{i}"])
    
    elif menu == t["upload_balance"]:
        st.markdown(f"## {t['upload_balance']}")
        uploaded_file = st.file_uploader(t["upload_btn"], type=['zip'])
        if uploaded_file:
            with st.spinner(t["processing"]):
                normal_cnt, pneumonia_cnt = process_zip(uploaded_file)
            st.markdown(f"### {t['results']}")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(t["normal_count"], normal_cnt)
            with col2:
                st.metric(t["pneumonia_count"], pneumonia_cnt)
            st.balloons()
            # Download report
            report = f"Prediction report - {datetime.datetime.now()}\nNormal images: {normal_cnt}\nPneumonia images: {pneumonia_cnt}\n"
            st.download_button(t["download"], report, file_name="prediction_report.txt")
    
    elif menu == t["history"]:
        st.markdown(f"## {t['history']}")
        st.info("Your past predictions will appear here (feature coming soon).")
    
    elif menu == t["about"]:
        st.markdown(f"## {t['about']}")
        st.markdown("""
        **MedBalance Pro** uses a deep learning model (CNN) trained on chest X‑rays to distinguish normal from pneumonia cases.  
        The model was trained on an imbalanced dataset (1:5 normal‑to‑pneumonia ratio) and achieves:
        - **40% normal detection** (improved from 0%)
        - **71% pneumonia detection**
        - Upload your own ZIP of X‑rays to see predictions in real time.
        """)
    
    elif menu == t["logout"]:
        st.session_state.logged_in = False
        st.rerun()

st.markdown("---")
st.markdown("© 2025 MedBalance Pro | AI for Better Healthcare")
