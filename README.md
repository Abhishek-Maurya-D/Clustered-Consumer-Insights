# 🛒 Clustered Consumer Insights

An interactive Machine Learning web application that segments customers into distinct shopping personalities. Built with Streamlit, this tool allows users to input their demographics and spending habits to see exactly where they fit within a complex marketplace.

## 🔗 Live Demo
Check out the live application here:  
**[SmartCart Insights · Streamlit](https://clustered-consumer-insights.streamlit.app/)**



## 🌟 Project Overview
This project analyzes raw customer data to discover hidden patterns in consumer behavior. It simplifies high-dimensional data so that a user can understand their "Group" without needing a background in data science. By entering details like income, age, and spending, the app "guesses" your persona using advanced mathematical models.

## 🛠️ Tech Stack
* **Frontend:** Streamlit (for the interactive Web UI)
* **Data Processing:** Pandas & NumPy
* **Machine Learning:** Scikit-Learn (StandardScaler, PCA, Agglomerative Clustering)
* **Mathematical Operations:** SciPy (Spatial distance calculation)
* **Visuals:** Matplotlib & Seaborn

## 🧠 Technical Highlights
The application follows a rigorous data science pipeline:
1.  **Comprehensive Inputs:** Collects data on Income, Recency, Age, Total Spending, and Family Size.
2.  **Preprocessing:** Handles missing values and filters outliers (Age < 90, Income < $600k).
3.  **Dimensionality Reduction:** Uses **Principal Component Analysis (PCA)** to condense 7+ variables into 3 core components for optimal clustering.
4.  **Clustering:** Employs **Agglomerative Hierarchical Clustering** with a 'ward' linkage to identify natural shopper archetypes.
5.  **Real-Time Inference:** Maps user input into the trained PCA space and identifies the nearest cluster center using Euclidean distance.

## 🚀 Installation & Usage
1. **Clone the Repo:**
   ```bash
   git clone [https://github.com/Abhishek-Maurya-D/Clustered-Consumer-Insights.git](https://github.com/Abhishek-Maurya-D/Clustered-Consumer-Insights.git)
   ```

2. **Install Requirements:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the App Locally:**
   ```bash
   streamlit run app.py
   ```



## 📊 Visualizing Your Persona

The app provides a real-time "Market Map." All existing customers are represented as light dots. When you enter your data, a **Red 'X'** appears, showing you exactly which "neighborhood" of shoppers you belong to—whether you are a *Premium Collector*, an *Essentialist*, or a *Family Provider*.
