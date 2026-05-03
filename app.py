# MedBalance Pro - Full Version with Dynamic Accuracy
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

# ============ AI MODEL ============
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
        st.warning("⚠️ Model file not found. Using random predictions for demo.")
        return None, device

model, device = load_model()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

def predict_image(image):
    if model is None:
        import random
        return random.randint(0, 1)
    img = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)
    return predicted.item()

def process_zip(zip_file):
    """Returns: normal_count, pneumonia_count, normal_accuracy, pneumonia_accuracy"""
    normal_count = 0
    pneumonia_count = 0
    correct_normal = 0
    correct_pneumonia = 0
    has_labels = False
    total_normal = 0
    total_pneumonia = 0
    
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_file, 'r') as zf:
            zf.extractall(tmpdir)
        
     # Check for labelled subfolders (case-insensitive)
normal_path = None
pneumonia_path = None

for item in os.listdir(tmpdir):
    if item.lower() == 'normal':
        normal_path = os.path.join(tmpdir, item)
    elif item.lower() == 'pneumonia':
        pneumonia_path = os.path.join(tmpdir, item)

if normal_path and pneumonia_path and os.path.isdir(normal_path) and os.path.isdir(pneumonia_path):
    has_labels = True
        
        # Scan all images
        for root, dirs, files in os.walk(tmpdir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        img_path = os.path.join(root, file)
                        img = Image.open(img_path)
                        pred = predict_image(img)  # 0=normal, 1=pneumonia
                        
                        if has_labels:
                            folder = os.path.basename(root)
                            if folder == 'normal':
                                total_normal += 1
                                normal_count += 1
                                if pred == 0:
                                    correct_normal += 1
                            elif folder == 'pneumonia':
                                total_pneumonia += 1
                                pneumonia_count += 1
                                if pred == 1:
                                    correct_pneumonia += 1
                        else:
                            # No labels – just count predictions
                            if pred == 0:
                                normal_count += 1
                            else:
                                pneumonia_count += 1
                    except Exception as e:
                        pass
    
    if has_labels:
        normal_accuracy = (correct_normal / total_normal * 100) if total_normal > 0 else 0
        pneumonia_accuracy = (correct_pneumonia / total_pneumonia * 100) if total_pneumonia > 0 else 0
        return normal_count, pneumonia_count, normal_accuracy, pneumonia_accuracy, total_normal, total_pneumonia
    else:
        return normal_count, pneumonia_count, None, None, 0, 0

# ============ MULTI-LANGUAGE TEXTS (simplified for brevity – add your full translations) ============
TEXTS = {
    "en": {
        "title": "🏥 MedBalance Pro",
        "subtitle": "Dynamic Medical Image Balancing Platform",
        "upload_btn": "Upload ZIP file (with normal/ and pneumonia/ folders for accuracy)",
        "processing": "Analyzing images with AI model...",
        "results": "Prediction Results",
        "normal_count": "Normal images",
        "pneumonia_count": "Pneumonia images",
        "normal_accuracy": "Normal detection accuracy",
        "pneumonia_accuracy": "Pneumonia detection accuracy",
        "download": "Download Report",
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
        "success": "Account created successfully!",
        "model_performance": "Model Performance (from validation set)",
        "normal_dynamic": "Dynamic Accuracy (from your upload)",
        "pneumonia_dynamic": "Dynamic Accuracy (from your upload)",
        "how_it_works": "How It Works",
        "step1": "1. For accuracy: upload a ZIP with 'normal' and 'pneumonia' folders",
        "step2": "2. For predictions only: upload any ZIP of X‑ray images",
        "step3": "3. AI model predicts each image",
        "step4": "4. Download full report"
    }
}

# ============ APP UI ============
st.set_page_config(page_title="MedBalance Pro", layout="wide")
st.title(TEXTS["en"]["title"])
st.markdown(f"### {TEXTS['en']['subtitle']}")

# Initialize session
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'user_id' not in st.session_state:
    st.session_state.user_id = None

# ============ LOGIN (simplified core) ============
if not st.session_state.logged_in:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Login")
        login_user = st.text_input("Username")
        login_pass = st.text_input("Password", type="password")
        if st.button("Login"):
            supabase = get_supabase()
            hashed = hash_password(login_pass)
            result = supabase.table("users").select("*").eq("username", login_user).eq("password", hashed).execute()
            if result.data:
                st.session_state.logged_in = True
                st.session_state.username = login_user
                st.session_state.user_id = result.data[0]['id']
                st.rerun()
            else:
                st.error("Invalid credentials")
    with col2:
        st.subheader("Sign Up")
        new_user = st.text_input("Username", key="su")
        new_pass = st.text_input("Password", type="password", key="sp")
        new_email = st.text_input("Email")
        if st.button("Create Account"):
            supabase = get_supabase()
            hashed = hash_password(new_pass)
            try:
                supabase.table("users").insert({"username": new_user, "password": hashed, "email": new_email}).execute()
                st.success("Account created! Please login.")
            except:
                st.error("Username already exists")
else:
    # ============ MAIN APP (LOGGED IN) ============
    st.sidebar.markdown(f"## 👋 {st.session_state.username}")
    menu = st.sidebar.radio("Menu", ["Dashboard", "Upload & Predict", "About", "Logout"])
    
    if menu == "Dashboard":
        st.subheader("Dashboard")
        st.markdown("### How to use dynamic accuracy")
        st.info("""
        **To get accuracy percentages:**
        - Upload a ZIP containing **two subfolders**:
          - `normal/` (put normal X‑rays here)
          - `pneumonia/` (put pneumonia X‑rays here)
        - The app will:
          1. Predict each image
          2. Compare predictions to true folder names
          3. Show **accuracy percentages** that change based on your data
        """)
        
        st.markdown("### Static model performance (from validation set)")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Normal detection (baseline)", "0%")
            st.metric("Normal detection (improved)", "40%", delta="+40%")
        with col2:
            st.metric("Pneumonia detection (baseline)", "100%")
            st.metric("Pneumonia detection (improved)", "71%", delta="-29%")
    
    elif menu == "Upload & Predict":
        st.subheader("Upload & Predict")
        st.write("Upload a ZIP file containing chest X‑ray images")
        st.caption("For accuracy, include 'normal' and 'pneumonia' subfolders. Otherwise, only prediction counts will be shown.")
        
        uploaded_file = st.file_uploader("Choose ZIP file", type=['zip'])
        
        if uploaded_file:
            with st.spinner("Processing images..."):
                result = process_zip(uploaded_file)
                normal_cnt, pneumonia_cnt, normal_acc, pneumonia_acc, total_normal, total_pneumonia = result
            
            st.success("✅ Processing complete!")
            st.subheader("Results")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Normal", normal_cnt)
                if normal_acc is not None:
                    st.metric("✅ Normal detection ACCURACY", f"{normal_acc:.1f}%", delta=f"on {total_normal} images")
            with col2:
                st.metric("Predicted Pneumonia", pneumonia_cnt)
                if pneumonia_acc is not None:
                    st.metric("✅ Pneumonia detection ACCURACY", f"{pneumonia_acc:.1f}%", delta=f"on {total_pneumonia} images")
            
            if normal_acc is None:
                st.info("ℹ️ No labelled subfolders found. Showing prediction counts only. To get accuracy, include 'normal' and 'pneumonia' folders in your ZIP.")
            else:
                st.balloons()
                st.success(f"🎯 Dynamic accuracy: Normal {normal_acc:.1f}% , Pneumonia {pneumonia_acc:.1f}%")
            
            # Download report
            report = f"MedBalance Report - {datetime.datetime.now()}\n"
            report += f"Normal images: {normal_cnt}\nPneumonia images: {pneumonia_cnt}\n"
            if normal_acc is not None:
                report += f"Normal accuracy: {normal_acc:.1f}%\nPneumonia accuracy: {pneumonia_acc:.1f}%"
            else:
                report += "No labelled data provided – accuracy not computed."
            st.download_button("📥 Download Report", report, file_name="medbalance_report.txt")
    
    elif menu == "About":
        st.subheader("About MedBalance Pro")
        st.markdown("""
        **Dynamic Accuracy Feature**
        - When you upload a ZIP with 'normal' and 'pneumonia' subfolders, the app computes **real accuracy** based on your images.
        - Accuracy percentages **change** depending on the quality and content of your uploaded data.
        
        **Model Performance (static from validation)**
        - Normal detection: 40%
        - Pneumonia detection: 71%
        
        **Technology**
        - Custom CNN (PyTorch)
        - Data augmentation + weighted loss
        - Deployed on Streamlit Cloud
        """)
    
    elif menu == "Logout":
        st.session_state.logged_in = False
        st.rerun()

st.markdown("---")
st.markdown("© 2025 MedBalance Pro | Dynamic Medical AI")
