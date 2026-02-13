import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist
from datetime import date

# --- STEP 1: LOAD & TRAIN (COMPREHENSIVE) ---
@st.cache_data
def load_and_train_model():
    # Load data from your repo
    url = "https://raw.githubusercontent.com/Abhishek-Maurya-D/Clustered-Consumer-Insights/main/smartcart_customers.csv"
    df = pd.read_csv(url)
    
    # Preprocessing matching your original script logic
    df["Income"] = df["Income"].fillna(df["Income"].median())
    df["Age"] = date.today().year - df["Year_Birth"]
    df["Total_Children"] = df["Kidhome"] + df["Teenhome"]
    df["Total_Spending"] = df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"] + \
                          df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"]
    
    df["Education"] = df["Education"].replace({
        "Basic": "Undergraduate", "2n Cycle": "Undergraduate",
        "Graduation": "Graduate", "Master": "Postgraduate", "PhD": "Postgraduate"
    })
    df["Living_with"] = df["Marital_Status"].replace({
        "Married": "Partner", "Together": "Partner", "Single": "Alone",
        "Divorced": "Alone", "Widow": "Alone", "Alone": "Alone", "YOLO": "Alone", "Absurd": "Alone"
    })

    # Features used for the final model
    features = ["Income", "Recency", "Age", "Total_Spending", "Total_Children", "Education", "Living_with"]
    df_model = df[features].copy()
    
    # Outlier Removal
    df_model = df_model[(df_model["Age"] < 90) & (df_model["Income"] < 600000)]
    
    # Encoding & Scaling
    ohe = OneHotEncoder(sparse_output=False)
    cat_enc = ohe.fit_transform(df_model[["Education", "Living_with"]])
    
    df_final = pd.concat([
        df_model.drop(columns=["Education", "Living_with"]).reset_index(drop=True),
        pd.DataFrame(cat_enc, columns=ohe.get_feature_names_out(["Education", "Living_with"]))
    ], axis=1)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_final)
    
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    
    # Using Agglomerative Clustering (4 clusters)
    model = AgglomerativeClustering(n_clusters=4)
    labels = model.fit_predict(X_pca)
    
    return scaler, ohe, pca, X_pca, labels, df_final

scaler, ohe, pca, X_pca, labels, df_final = load_and_train_model()

# --- STEP 2: USER INTERFACE ---
st.set_page_config(page_title="SmartCart Insights", layout="wide")
st.title("🛒 SmartCart Persona Finder")
st.write("Complete your profile below to see your customer segment.")

with st.sidebar:
    st.header("👤 Personal Details")
    age = st.slider("Age", 18, 90, 30)
    edu = st.selectbox("Education Level", ["Undergraduate", "Graduate", "Postgraduate"])
    living = st.selectbox("Living Situation", ["Partner", "Alone"])
    kids = st.number_input("Number of Children/Teens at home", 0, 5, 0)
    
    st.header("💰 Spending & Habits")
    income = st.number_input("Annual Income ($)", 0, 600000, 45000)
    recency = st.slider("Days since last shop", 0, 100, 10)
    
    st.subheader("Monthly Spend on:")
    wines = st.number_input("Wines ($)", 0, 2000, 100)
    meat = st.number_input("Meat ($)", 0, 2000, 100)
    other = st.number_input("Other (Fruit, Fish, Sweets, Gold) ($)", 0, 2000, 50)
    total_spend = wines + meat + other

# --- STEP 3: CALCULATION ---
# Match the user input to the dataframe structure
user_input = pd.DataFrame([[income, recency, age, total_spend, kids, edu, living]], 
                          columns=["Income", "Recency", "Age", "Total_Spending", "Total_Children", "Education", "Living_with"])

# Encode & Scale
user_cat = ohe.transform(user_input[["Education", "Living_with"]])
user_num = user_input.drop(columns=["Education", "Living_with"])
user_combined = pd.concat([user_num, pd.DataFrame(user_cat, columns=ohe.get_feature_names_out(["Education", "Living_with"]))], axis=1)

user_scaled = scaler.transform(user_combined)
user_pca = pca.transform(user_scaled)

# Determine Cluster based on proximity to means
cluster_centers = np.array([X_pca[labels == i].mean(axis=0) for i in range(4)])
user_cluster = cdist(user_pca, cluster_centers).argmin()

# --- STEP 4: RESULTS DISPLAY ---
st.divider()
c1, c2 = st.columns([1, 1])

with c1:
    st.header(f"You belong to: Group {user_cluster}")
    
    personalities = {
        0: "🌟 **The Premium Collector**: You prioritize quality and enjoy the finer things (like wines and meats). You are a frequent and highly valued shopper.",
        1: "🛒 **The Essentialist**: You shop when you need to and focus on practicality. You represent the core reliable customer base.",
        2: "👨‍👩‍👧 **The Family Provider**: Your spending is balanced across all categories, likely focused on household needs and family stability.",
        3: "📉 **The Selective Shopper**: You are very careful with your budget or perhaps new to our store. You look for specific deals."
    }
    st.success(personalities[user_cluster])
    
    st.metric("Total Spending", f"${total_spend}")
    st.metric("Estimated Market Position", f"Group {user_cluster}")

with c2:
    st.subheader("Your Position in our Market")
    
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette="viridis", alpha=0.3, ax=ax, s=15)
    plt.scatter(user_pca[0, 0], user_pca[0, 1], c='red', marker='X', s=250, label='YOU')
    ax.set_title("How you compare to other customers")
    ax.set_xlabel("Spending Power")
    ax.set_ylabel("Shopping Frequency")
    st.pyplot(fig)

st.info("💡 **Tech Note:** The graph uses PCA to turn 7 variables into a 2D map. Your position (Red X) is calculated using the distance to the nearest group center.")
