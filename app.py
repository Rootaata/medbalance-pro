# MedBalance Pro – ResNet‑18 Model (downloaded from Google Drive)
import streamlit as st
import hashlib
from supabase import create_client
import zipfile
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import tempfile
import datetime
import random
import gdown

# ============ SUPABASE ============
SUPABASE_URL = "https://uhskvktshxojggmfcqtd.supabase.co"
SUPABASE_KEY = "sb_publishable_ArKrKiXzj4Wq04-9N-dMBA_iug1h2hm"

def get_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def hash_password(pwd):
    return hashlib.sha256(pwd.encode()).hexdigest()

# ============ DOWNLOAD RESNET‑18 MODEL FROM GOOGLE DRIVE ============
@st.cache_resource
def load_model():
    device = torch.device('cpu')
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(512, 2)
    model_path = 'resnet18_best.pth'
    
    # Download if not exists
    if not os.path.exists(model_path):
        with st.spinner("Downloading model from Google Drive (50 MB)... Please wait."):
            file_id = "1zOzhGbbEdZxPa5IrKnL48kai-alkJrVY"
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, model_path, quiet=False)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

model, device = load_model()

# ResNet-18 expects 3-channel 224x224 images
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_image(image):
    img = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(img)
        _, pred = torch.max(out, 1)
    return pred.item()

def process_zip_with_accuracy(zip_file):
    normal_pred = 0
    pneumonia_pred = 0
    total_normal = 0
    total_pneumonia = 0
    correct_normal = 0
    correct_pneumonia = 0
    has_labels = False

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_file, 'r') as zf:
            zf.extractall(tmpdir)

        normal_folder = None
        pneumonia_folder = None
        for root, dirs, files in os.walk(tmpdir):
            for d in dirs:
                if d.lower() == 'normal':
                    normal_folder = os.path.join(root, d)
                elif d.lower() == 'pneumonia':
                    pneumonia_folder = os.path.join(root, d)

        if normal_folder and pneumonia_folder:
            has_labels = True

        for root, dirs, files in os.walk(tmpdir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        img = Image.open(os.path.join(root, file))
                        pred = predict_image(img)
                        if has_labels:
                            if normal_folder and os.path.join(root, file).startswith(normal_folder):
                                total_normal += 1
                                normal_pred += 1
                                if pred == 0:
                                    correct_normal += 1
                            elif pneumonia_folder and os.path.join(root, file).startswith(pneumonia_folder):
                                total_pneumonia += 1
                                pneumonia_pred += 1
                                if pred == 1:
                                    correct_pneumonia += 1
                        else:
                            if pred == 0:
                                normal_pred += 1
                            else:
                                pneumonia_pred += 1
                    except:
                        pass

    if has_labels:
        normal_acc = (correct_normal / total_normal * 100) if total_normal > 0 else 0
        pneumonia_acc = (correct_pneumonia / total_pneumonia * 100) if total_pneumonia > 0 else 0
        return normal_pred, pneumonia_pred, normal_acc, pneumonia_acc
    else:
        return normal_pred, pneumonia_pred, None, None

# ============ UI ============
st.set_page_config(page_title="MedBalance Pro", layout="wide")
st.title("🏥 MedBalance Pro")
st.markdown("### Medical Image Balancing Platform (ResNet‑18 Model)")

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None

# ---------- LOGIN / SIGNUP ----------
if not st.session_state.logged_in:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Login")
        user = st.text_input("Username")
        pwd = st.text_input("Password", type="password")
        if st.button("Login"):
            supabase = get_supabase()
            hashed = hash_password(pwd)
            res = supabase.table("users").select("*").eq("username", user).eq("password", hashed).execute()
            if res.data:
                st.session_state.logged_in = True
                st.session_state.username = user
                st.rerun()
            else:
                st.error("Invalid credentials")
    with col2:
        st.subheader("Sign Up")
        new_user = st.text_input("Username", key="su")
        new_pwd = st.text_input("Password", type="password", key="sp")
        new_email = st.text_input("Email")
        if st.button("Create Account"):
            supabase = get_supabase()
            hashed = hash_password(new_pwd)
            try:
                supabase.table("users").insert({"username": new_user, "password": hashed, "email": new_email}).execute()
                st.success("Account created! Please login.")
            except:
                st.error("Username exists")
else:
    st.sidebar.markdown(f"## 👋 {st.session_state.username}")
    menu = st.sidebar.radio("Menu", ["Dashboard", "Upload & Predict", "About", "Logout"])

    if menu == "Dashboard":
        st.subheader("Model Performance (ResNet‑18)")
        st.markdown("""
        | Configuration | Normal Detection | Pneumonia Detection |
        |---------------|------------------|----------------------|
        | Baseline (no balancing) | 0% | 100% |
        | Custom CNN + MixUp | 100% | 0% |
        | **ResNet‑18 + MixUp (this app)** | **100%** | **91%** |
        """)
        st.info("Upload a ZIP with `normal` and `pneumonia` folders to test dynamic accuracy.")

    elif menu == "Upload & Predict":
        st.subheader("Upload & Predict (Dynamic Accuracy)")
        st.write("Upload a ZIP file containing chest X‑ray images.")
        st.caption("For accuracy, include folders named `normal` and `pneumonia` (case‑insensitive).")
        uploaded = st.file_uploader("Choose ZIP file", type=['zip'])
        if uploaded:
            with st.spinner("Processing..."):
                norm_pred, pneum_pred, norm_acc, pneum_acc = process_zip_with_accuracy(uploaded)
            st.success("Done!")
            col1, col2 = st.columns(2)
            col1.metric("Predicted Normal", norm_pred)
            col2.metric("Predicted Pneumonia", pneum_pred)
            if norm_acc is not None:
                st.subheader("📊 Dynamic Accuracy")
                col1, col2 = st.columns(2)
                col1.metric("Normal Accuracy", f"{norm_acc:.1f}%")
                col2.metric("Pneumonia Accuracy", f"{pneum_acc:.1f}%")
                st.balloons()
            else:
                st.info("No labelled subfolders found. Include 'normal' and 'pneumonia' folders.")
            report = f"Report {datetime.datetime.now()}\nNormal: {norm_pred}\nPneumonia: {pneum_pred}"
            if norm_acc is not None:
                report += f"\nNormal accuracy: {norm_acc:.1f}%\nPneumonia accuracy: {pneum_acc:.1f}%"
            st.download_button("Download Report", report, file_name="report.txt")

    elif menu == "About":
        st.markdown("""
        **Model:** ResNet‑18 trained with augmentation, weighted loss, and MixUp synthetic data.  
        **Performance on test set:** 100% normal recall, 91% pneumonia recall.  
        **Upload labelled data to see dynamic accuracy.**
        """)
    elif menu == "Logout":
        st.session_state.logged_in = False
        st.rerun()

st.markdown("---")
st.markdown("© 2025 MedBalance Pro | ResNet‑18 Model")
