import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from fpdf import FPDF

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Hospital AI System", layout="wide")

# -----------------------------
# LOGIN SYSTEM
# -----------------------------
if "login" not in st.session_state:
    st.session_state.login = False

if not st.session_state.login:
    st.title("🔐 Hospital Login")

    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    st.info("Use: admin / 1234")

    if st.button("Login"):
        if u.strip() == "admin" and p.strip() == "1234":
            st.session_state.login = True
            st.success("Login Successful")
            st.rerun()
        else:
            st.error("Invalid Login")

    st.stop()

# -----------------------------
# SIDEBAR UI
# -----------------------------
st.sidebar.markdown("## 🏥 Hospital AI System")

st.sidebar.markdown("""
✨ *Smart Healthcare Assistant*  

- 🤖 AI Predictions  
- 📊 Real-time Learning  
- 🧠 Intelligent Diagnosis  
""")

st.sidebar.divider()

menu = st.sidebar.radio("📌 Navigation", ["🏠 Dashboard", "🧪 Prediction", "📂 Dataset", "📊 Charts"])

disease = st.sidebar.selectbox("🩺 Select Disease", ["Diabetes", "Heart", "Parkinson"])

st.sidebar.divider()

st.sidebar.info("💡 Early detection saves lives!")

st.sidebar.markdown("---")
st.sidebar.markdown("👩‍💻 Developed by Anjali Mehra")

# -----------------------------
# FILE PATHS
# -----------------------------
files = {
    "Diabetes": "diabetes.csv",
    "Heart": "heart.csv",
    "Parkinson": "parkinson.csv"
}

columns_dict = {
    "Diabetes": ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DPF","Age","Outcome"],
    "Heart": ["Age","Sex","ChestPain","BP","Cholesterol","FBS","ECG","MaxHR","Angina","Oldpeak","Slope","Outcome"],
    "Parkinson": ["Fo","Fhi","Flo","Jitter","Shimmer","NHR","HNR","RPDE","DFA","Spread1","Spread2","D2","PPE","Outcome"]
}

# -----------------------------
# LOAD DATA
# -----------------------------
def load_data(disease):
    file = files[disease]
    cols = columns_dict[disease]

    if not os.path.exists(file):
        df = pd.DataFrame(columns=cols)
        df.to_csv(file, index=False)

    df = pd.read_csv(file)

    # Clean data
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(df.mean())

    return df

# -----------------------------
# SAVE DATA
# -----------------------------
def save_data(disease, row):
    file = files[disease]
    df = pd.read_csv(file)

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(file, index=False)

# -----------------------------
# TRAIN MODEL (FIXED)
# -----------------------------
def train_model(df):

    df = df.copy()

    # CLEAN DATA
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(df.mean())
    df = df.dropna()

    if len(df) < 5:
        return None, None

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    return model, scaler

# -----------------------------
# PDF FUNCTION
# -----------------------------
def create_pdf(disease, result):
    clean_result = result.encode('latin-1', 'ignore').decode('latin-1')

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)

    pdf.cell(200, 10, "Hospital AI Report", ln=True)
    pdf.cell(200, 10, f"Disease: {disease}", ln=True)
    pdf.cell(200, 10, f"Result: {clean_result}", ln=True)

    return pdf.output(dest='S').encode('latin-1')

# -----------------------------
# DASHBOARD
# -----------------------------
if menu == "🏠 Dashboard":
    st.title("🏥 Hospital Dashboard")

    df = load_data(disease)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records", len(df))
    c2.metric("Positive Cases", int(df["Outcome"].sum()) if len(df)>0 else 0)
    c3.metric("System Status", "Active")

# -----------------------------
# PREDICTION
# -----------------------------
elif menu == "🧪 Prediction":

    st.title(f"{disease} Prediction")

    df = load_data(disease)

    inputs = []
    for col in columns_dict[disease][:-1]:
        val = st.number_input(col, value=0.0)
        inputs.append(val)

    outcome = st.selectbox("Actual Condition (Training)", [0, 1])

    if st.button("Predict & Save"):

        # SAVE DATA
        row = inputs + [outcome]
        save_data(disease, row)

        df = load_data(disease)
        model, scaler = train_model(df)

        if model is None:
            st.warning("⚠ Add at least 5 records")
        else:
            pred = model.predict(scaler.transform([inputs]))

            if pred[0] == 1:
                if disease == "Diabetes":
                    result = "🩸 Diabetic"
                elif disease == "Heart":
                    result = "❤️ Heart Disease"
                else:
                    result = "🧠 Parkinson"
                st.error(result)
            else:
                if disease == "Diabetes":
                    result = "✅ Non-Diabetic"
                elif disease == "Heart":
                    result = "💚 Healthy"
                else:
                    result = "✅ Normal"
                st.success(result)

            pdf = create_pdf(disease, result)
            st.download_button("📄 Download Report", pdf, "report.pdf")

# -----------------------------
# DATASET VIEW
# -----------------------------
elif menu == "📂 Dataset":
    st.title("📂 Dataset Viewer")

    df = load_data(disease)
    st.dataframe(df)

# -----------------------------
# CHARTS
# -----------------------------
elif menu == "📊 Charts":
    st.title("📊 Analytics")

    df = load_data(disease)

    if len(df) > 0:
        col = st.selectbox("Select Column", df.columns)
        st.line_chart(df[col])
    else:
        st.warning("No data available")