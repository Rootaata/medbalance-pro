# MedBalance Pro - Final Working Version (Dynamic Accuracy)
import streamlit as st
import hashlib
import pandas as pd
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

# ============ SUPABASE ============
SUPABASE_URL = "https://uhskvktshxojggmfcqtd.supabase.co"
SUPABASE_KEY = "sb_publishable_ArKrKiXzj4Wq04-9N-dMBA_iug1h2hm"

def get_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def hash_password(pwd):
    return hashlib.sha256(pwd.encode()).hexdigest()

# ============ MODEL ============
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
        st.warning("Model file not found. Using random predictions.")
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

def process_zip(zip_file, user_id=None):
    # Returns (normal_pred, pneumonia_pred, normal_acc, pneumonia_acc, total_normal, total_pneumonia)
    normal_pred = 0
    pneumonia_pred = 0
    correct_normal = 0
    correct_pneumonia = 0
    total_normal = 0
    total_pneumonia = 0
    has_labels = False

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_file, 'r') as zf:
            zf.extractall(tmpdir)

        # Recursively find all folders named 'normal' or 'pneumonia' (case-insensitive)
        normal_folders = []
        pneumonia_folders = []
        for root, dirs, files in os.walk(tmpdir):
            for d in dirs:
                if d.lower() == 'normal':
                    normal_folders.append(os.path.join(root, d))
                elif d.lower() == 'pneumonia':
                    pneumonia_folders.append(os.path.join(root, d))

        if normal_folders and pneumonia_folders:
            has_labels = True
            # Use the first found normal and pneumonia folders
            normal_path = normal_folders[0]
            pneumonia_path = pneumonia_folders[0]
        else:
            has_labels = False

        # Process images
        for root, dirs, files in os.walk(tmpdir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        img_path = os.path.join(root, file)
                        img = Image.open(img_path)
                        pred = predict_image(img)  # 0=normal, 1=pneumonia

                        if has_labels:
                            # Determine truth by checking if the image is inside a normal folder
                            is_in_normal = any(img_path.startswith(folder) for folder in normal_folders)
                            is_in_pneumonia = any(img_path.startswith(folder) for folder in pneumonia_folders)
                            if is_in_normal:
                                total_normal += 1
                                normal_pred += 1
                                if pred == 0:
                                    correct_normal += 1
                            elif is_in_pneumonia:
                                total_pneumonia += 1
                                pneumonia_pred += 1
                                if pred == 1:
                                    correct_pneumonia += 1
                        else:
                            # Unlabelled: just count predictions
                            if pred == 0:
                                normal_pred += 1
                            else:
                                pneumonia_pred += 1
                    except Exception as e:
                        pass

    if has_labels:
        normal_acc = (correct_normal / total_normal * 100) if total_normal > 0 else 0
        pneumonia_acc = (correct_pneumonia / total_pneumonia * 100) if total_pneumonia > 0 else 0
        # Save history
        if user_id:
            try:
                supabase = get_supabase()
                supabase.table("history").insert({
                    "user_id": user_id,
                    "normal_count": normal_pred,
                    "pneumonia_count": pneumonia_pred,
                    "normal_accuracy": normal_acc,
                    "pneumonia_accuracy": pneumonia_acc
                }).execute()
            except:
                pass
        return normal_pred, pneumonia_pred, normal_acc, pneumonia_acc, total_normal, total_pneumonia
    else:
        return normal_pred, pneumonia_pred, None, None, 0, 0

# ============ UI ============
st.set_page_config(page_title="MedBalance Pro", layout="wide")
st.title("MedBalance Pro")
st.markdown("### Dynamic Medical Image Balancing Platform")

# Session
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'user_id' not in st.session_state:
    st.session_state.user_id = None

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
    st.sidebar.markdown(f"## 👋 {st.session_state.username}")
    menu = st.sidebar.radio("Menu", ["Dashboard", "Upload & Predict", "About", "Logout"])

    if menu == "Dashboard":
        st.info("Upload a ZIP containing `normal` and `pneumonia` folders (any case, any nesting level) to get dynamic accuracy.")
        st.markdown("**Model validation performance (static)**")
        col1, col2 = st.columns(2)
        col1.metric("Normal (baseline)", "0%")
        col1.metric("Normal (balanced)", "40%", delta="+40%")
        col2.metric("Pneumonia (baseline)", "100%")
        col2.metric("Pneumonia (balanced)", "71%", delta="-29%")

    elif menu == "Upload & Predict":
        st.subheader("Upload & Predict")
        uploaded = st.file_uploader("Choose ZIP", type=['zip'])
        if uploaded:
            with st.spinner("Processing..."):
                normal_pred, pneumonia_pred, acc_n, acc_p, total_n, total_p = process_zip(uploaded, st.session_state.user_id)
            st.success("Done!")
            col1, col2 = st.columns(2)
            col1.metric("Predicted Normal", normal_pred)
            col2.metric("Predicted Pneumonia", pneumonia_pred)
            if acc_n is not None:
                st.subheader("Dynamic Accuracy (based on your uploaded labels)")
                col1, col2 = st.columns(2)
                col1.metric("Normal accuracy", f"{acc_n:.1f}%", delta=f"on {total_n} images")
                col2.metric("Pneumonia accuracy", f"{acc_p:.1f}%", delta=f"on {total_p} images")
                st.balloons()
            else:
                st.info("No labelled folders found. Only prediction counts shown. To get accuracy, include folders named 'normal' and 'pneumonia' (case-insensitive) in the ZIP.")

            # Download report
            report = f"Report {datetime.datetime.now()}\nPredicted Normal: {normal_pred}\nPredicted Pneumonia: {pneumonia_pred}\n"
            if acc_n is not None:
                report += f"Normal accuracy: {acc_n:.1f}% ({total_n} images)\nPneumonia accuracy: {acc_p:.1f}% ({total_p} images)"
            else:
                report += "No labelled data – accuracy not computed."
            st.download_button("Download Report", report, file_name="medbalance_report.txt")

    elif menu == "About":
        st.markdown("""
        **Dynamic Accuracy** works when you upload a ZIP that contains at least one folder named `normal` (case-insensitive) and one named `pneumonia`.  
        The app will compare its predictions to the folder names and show accuracy percentages that change with your data.
        """)

    elif menu == "Logout":
        st.session_state.logged_in = False
        st.rerun()

st.markdown("---")
st.markdown("© 2025 MedBalance Pro")
