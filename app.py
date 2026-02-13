# import libraries and modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.cluster import AgglomerativeClustering

#reading csv file
df = pd.read_csv("smartcart_customers.csv")

# filling the missing income with median values
df["Income"] = df["Income"].fillna(df["Income"].median())

# Get the current year
current_year = date.today().year

# calculating the age of the users
df["Age"] = current_year - df["Year_Birth"]

# Customer's Joining Date
df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True)
reference_date = df["Dt_Customer"].max()
df["Customer_Tenure_Days"] = (reference_date - df["Dt_Customer"]).dt.days

# Spending -> Total
df["Total_Spending"] = df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"] + df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"];

# Children
df["Total_Children"] = df["Kidhome"] + df["Teenhome"];

# Education
df["Education"] = df["Education"].replace({
    "Basic": "Undergraduate",
    "2n Cycle": "Undergraduate",
    "Graduation": "Graduate",
    "Master": "Postgraduate",
    "PhD": "Postgraduate"
})

# Marital Status
df["Marital_Status"].value_counts()

# Living Status
df["Living_with"] = df["Marital_Status"].replace({
    "Married": "Partner",
    "Together": "Partner",
    "Single": "Alone",
    "Divorced": "Alone",
    "Widow": "Alone",
    "Alone": "Alone",
    "YOLO": "Alone",
    "Absurd": "Alone"
})

# droping unneccessary columns
cols = ["ID", "Year_Birth", "Marital_Status", "Kidhome", "Teenhome", "Dt_Customer"]
spending_cols = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
cols_to_drop = cols + spending_cols;
df_cleaned = df.drop(columns = cols_to_drop)

# outliers -> deleting
cols = ["Income", "Recency", "Response", "Age", "Total_Spending", "Total_Children"];

# visualing outliers
# sns.pairplot(df_cleaned[cols])

# there are some outliers 
# 1. having salary around 600000
# 2. age above and around 120
print("data size with outliers", len(df_cleaned))
df_cleaned = df_cleaned[(df_cleaned["Age"] < 90)]
df_cleaned = df_cleaned[(df_cleaned["Income"] < 600000)]
print("data size without outliers", len(df_cleaned))

# checking the correleation of different features -> using heatmaps
corr = df_cleaned.corr(numeric_only=True)

# remove comments for visualisation
# plt.figure(figsize=(8,6))
# sns.heatmap(
#     corr, 
#     annot=True,
#     annot_kws = {"size":6},
#     cmap = "coolwarm"
# )

# Encoding
ohe = OneHotEncoder()
cat_cols = ["Education", "Living_with"]
enc_cols = ohe.fit_transform(df_cleaned[cat_cols])
enc_df = pd.DataFrame(enc_cols.toarray(), columns=ohe.get_feature_names_out(cat_cols), index=df_cleaned.index)
df_encoded = pd.concat([df_cleaned.drop(columns=cat_cols), enc_df], axis=1)

# scaling
X = df_encoded
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3D Visualisation
pca = PCA(n_components = 3)
X_pca = pca.fit_transform(X_scaled)

# Analysising value of k
wcss = []
for k in range(1,11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit_transform(X_pca)
    wcss.append(kmeans.inertia_)
knee = KneeLocator(range(1,11), wcss, curve="convex", direction="decreasing")
optimal_k = knee.elbow

# clustering
# Agglomerative Clustering
agg_clf = AgglomerativeClustering(n_clusters=optimal_k, linkage="ward")
labels_agg = agg_clf.fit_predict(X_pca)

# Plot
# fig = plt.figure(figsize=(10,8))
# ax = fig.add_subplot(111, projection="3d")
# ax.scatter(X_pca[:,0], X_pca[:,1], X_pca[:,2], c=labels_agg)

# characterisation of clusters
X["cluster"] = labels_agg
pal = ["red", "blue", "yellow", "green"]
cluster_summary = X.groupby("cluster").mean()
print(cluster_summary)
