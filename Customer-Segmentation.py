import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans

# Load dataset
data = pd.read_csv(r"C:\Users\bashe\Downloads\archive (3)\Mall_Customers.csv")

# Select relevant columns
X = data[["Genre", "Age", "Annual Income (k$)", "Spending Score (1-100)"]]
label = LabelEncoder()
X["Genre"] = label.fit_transform(X["Genre"])

# Scale numerical features
scaler = StandardScaler()
X[["Age", "Annual Income (k$)", "Spending Score (1-100)"]] = scaler.fit_transform(
    X[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
)

# Apply K-Means
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
kmeans.fit(X)
X["Cluster"] = kmeans.labels_

# Visualization
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=X["Annual Income (k$)"],
    y=X["Spending Score (1-100)"],
    hue=X["Cluster"],
    palette="Set2",
    s=100
)
plt.title("Customer Clusters (Annual Income vs Spending Score)")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend(title="Cluster")
plt.show()

# Add clusters back to original data
data["Cluster"] = X["Cluster"]

# Define human-readable labels
cluster_labels = {
    0: "Mature Average Spenders",
    1: "Young Luxury Shoppers",
    2: "Young Budget Shoppers",
    3: "Wealthy Low Spenders"
}

# Second scatter plot with color-coded clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    data=data,
    palette="set2",
    s=100
)
plt.title("Customer Segmentation using K-Means")
plt.show()

# Take new user input
genre_input = input("Enter your Genre (Male/Female): ").strip().capitalize()
age_input = int(input("Enter your Age: "))
income_input = int(input("Enter your Annual Income (k$): "))
spending_input = int(input("Enter your Spending Score (1-100): "))

# Prepare new data
new_data = pd.DataFrame({
    "Genre": [genre_input],
    "Age": [age_input],
    "Annual Income (k$)": [income_input],
    "Spending Score (1-100)": [spending_input]
})

# Encode and scale new input
new_data["Genre"] = label.transform(new_data["Genre"])
new_data[["Age", "Annual Income (k$)", "Spending Score (1-100)"]] = scaler.transform(
    new_data[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
)

# Predict cluster
cluster = kmeans.predict(new_data)[0]
print(f"\nThis customer belongs to Cluster {cluster}: {cluster_labels[cluster]}")

# Cluster summary
cluster_summary = data.groupby("Cluster").mean(numeric_only=True)
print("\nCluster Summary:\n", cluster_summary)