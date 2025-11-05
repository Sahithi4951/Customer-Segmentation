# 🛍️ Customer Segmentation using K-Means Clustering

This project uses **K-Means Clustering** to segment mall customers based on their **Age**, **Gender**, **Annual Income**, and **Spending Score**.  
It helps businesses understand different customer groups and tailor marketing strategies accordingly.

---

## 🚀 Features

- Uses the **Mall Customers Dataset**
- Performs **data preprocessing** (Label Encoding + Standard Scaling)
- Applies the **Elbow Method** to find the optimal number of clusters
- Visualizes customer groups using **Seaborn**
- Allows user input to **predict which cluster** a new customer belongs to
- Summarizes cluster characteristics (e.g., Luxury, Average, etc.)

---

## 📂 Dataset

The dataset used:  
`Mall_Customers.csv`

Columns:
- `CustomerID`
- `Genre`
- `Age`
- `Annual Income (k$)`
- `Spending Score (1-100)`

---

## 🧠 Model Workflow

1. Load and preprocess the dataset  
2. Encode categorical variables (`Genre`)  
3. Scale numerical features (`Age`, `Annual Income`, `Spending Score`)  
4. Use the **Elbow Method** to determine optimal `k`  
5. Apply **K-Means Clustering**  
6. Visualize clusters  
7. Predict the cluster for a new customer input  

---

## 📊 Cluster Interpretation

| Cluster | Description |
|----------|--------------|
| 0 | Older Average Spenders |
| 1 | Young Luxury Shoppers |
| 2 | Young Average Shoppers |
| 3 | High-Income Low Spenders |

---

## 💻 Run the Project

```bash
pip install numpy pandas seaborn matplotlib scikit-learn
python main.py

## 💬 Example Prediction

Enter your Genre (Male/Female):  Male
Enter your Age:  40
Enter your Annual Income (k$):  40
Enter your Spending Score (1-100):  30
This customer belongs to Cluster 0: Mature Average Spenders
Cluster Summary:
          CustomerID        Age  Annual Income (k$)  Spending Score (1-100)
Cluster                                                                   
0         68.484375  53.906250           47.343750               40.421875
1        161.025000  32.875000           86.100000               81.525000
2         53.438596  25.438596           40.000000               60.298246
3        159.743590  39.871795           86.102564               19.358974

---

## 📈 Results

![Customer Clusters (Annual Income vs Spending Score)](Customer Clusters (Annual Income vs Spending Score).png)
![Customer Segmentation using K-Means](Customer Segmentation using K-Means.png)

---

## 🧰 Libraries Used

numpy
pandas
seaborn
matplotlib
scikit-learn

---

## ✨ Author

Sahithi Bashetty
📧 bashettysahithi@gmail.com