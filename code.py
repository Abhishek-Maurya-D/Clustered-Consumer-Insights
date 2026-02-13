import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from datetime import date

# --- STEP 1: LOAD & PREPARE MODEL DATA ---
@st.cache_data
def load_and_train_model():
    # Load your existing data to "train" the segments
    df = pd.read_csv("smartcart_customers.csv")
    df["Income"] = df["Income"].fillna(df["Income"].median())
    df["Age"] = date.today().year - df["Year_Birth"]
    
    # Simple Pre-processing
    df["Education"] = df["Education"].replace({
        "Basic": "Undergraduate", "2n Cycle": "Undergraduate",
        "Graduation": "Graduate", "Master": "Postgraduate", "PhD": "Postgraduate"
    })
    df["Living_with"] = df["Marital_Status"].replace({
        "Married": "Partner", "Together": "Partner",
        "Single": "Alone", "Divorced": "Alone", "Widow": "Alone",
        "Alone": "Alone", "YOLO": "Alone", "Absurd": "Alone"
    })
    
    # Feature selection (matching your script)
    features = ["Income", "Recency", "Age", "Education", "Living_with"]
    df_model = df[features].copy()
    
    # Encoding & Scaling
    ohe = OneHotEncoder(sparse_output=False)
    cat_enc = ohe.fit_transform(df_model[["Education", "Living_with"]])
    cat_cols = ohe.get_feature_names_out(["Education", "Living_with"])
    
    df_final = pd.concat([
        df_model.drop(columns=["Education", "Living_with"]).reset_index(drop=True),
        pd.DataFrame(cat_enc, columns=cat_cols)
    ], axis=1)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_final)
    
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    
    # Clustering (Using 4 as a standard based on your palette)
    model = AgglomerativeClustering(n_clusters=4)
    labels = model.fit_predict(X_pca)
    
    return scaler, ohe, pca, model, df_final, X_pca, labels

scaler, ohe, pca, agg_model, df_final, X_pca, labels = load_and_train_model()

# --- STEP 2: USER INTERFACE ---
st.title("🛍️ SmartCart Customer Segment Finder")
st.markdown("Enter your details below to see which shopping personality group you belong to!")

with st.sidebar:
    st.header("Your Profile")
    age = st.slider("Your Age", 18, 90, 35)
    income = st.number_input("Annual Income ($)", 0, 200000, 50000)
    recency = st.slider("Days since last purchase", 0, 100, 20)
    edu = st.selectbox("Education Level", ["Undergraduate", "Graduate", "Postgraduate"])
    living = st.selectbox("Living Status", ["Partner", "Alone"])

# --- STEP 3: PREDICTION LOGIC ---
# Create a dataframe for the new user
user_data = pd.DataFrame([[income, recency, age, edu, living]], 
                         columns=["Income", "Recency", "Age", "Education", "Living_with"])

# Transform user data to match training data
user_cat_enc = ohe.transform(user_data[["Education", "Living_with"]])
user_final = pd.concat([
    user_data.drop(columns=["Education", "Living_with"]),
    pd.DataFrame(user_cat_enc, columns=ohe.get_feature_names_out(["Education", "Living_with"]))
], axis=1)

# Note: AgglomerativeClustering doesn't have .predict(), 
# so we find the closest cluster in PCA space for the user
user_scaled = scaler.transform(user_final)
user_pca = pca.transform(user_scaled)

# Finding the nearest cluster mean
from scipy.spatial.distance import cdist
cluster_centers = np.array([X_pca[labels == i].mean(axis=0) for i in range(4)])
user_cluster = cdist(user_pca, cluster_centers).argmin()

# --- STEP 4: DISPLAY RESULTS ---
st.divider()
col1, col2 = st.columns([1, 1])

with col1:
    st.header(f"You are in Group {user_cluster}")
    
    # Friendly Descriptions
    descriptions = {
        0: "**The High-Value Shopper**: You enjoy premium products and shop frequently.",
        1: "**The Budget-Conscious Explorer**: You look for deals and shop occasionally.",
        2: "**The Loyal Family Shopper**: You balance quality with family needs.",
        3: "**The Occasional Spender**: You shop mainly for essentials or during big sales."
    }
    st.info(descriptions.get(user_cluster, "Unique Shopper Profile"))

with col2:
    # Visualization
    fig = plt.figure(figsize=(6, 4))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.2, s=10)
    plt.scatter(user_pca[0, 0], user_pca[0, 1], c='red', marker='X', s=200, label='YOU')
    plt.title("Where you fit among our customers")
    plt.xlabel("Spending Habits (PCA1)")
    plt.ylabel("Lifestyle Factor (PCA2)")
    plt.legend()
    st.pyplot(fig)

st.success("This visualization shows all our customers as dots. The Red 'X' is you!")
