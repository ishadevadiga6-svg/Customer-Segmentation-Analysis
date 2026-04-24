# ── Step 1: Import Libraries ──────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ── Step 2: Load Dataset ──────────────────────────────
df = pd.read_csv('Mall_Customers.csv')
print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nBasic Statistics:")
print(df.describe())

# ── Step 3: Data Cleaning ─────────────────────────────
print("\nMissing Values:")
print(df.isnull().sum())

# Rename columns for easier use
df.columns = ['CustomerID', 'Gender', 'Age', 'AnnualIncome', 'SpendingScore']
print("\nCleaned Column Names:", df.columns.tolist())

# ── Step 4: Exploratory Data Analysis ────────────────
plt.figure(figsize=(15, 5))

# Age Distribution
plt.subplot(1, 3, 1)
sns.histplot(df['Age'], bins=20, color='steelblue', kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')

# Annual Income Distribution
plt.subplot(1, 3, 2)
sns.histplot(df['AnnualIncome'], bins=20, color='coral', kde=True)
plt.title('Annual Income Distribution')
plt.xlabel('Annual Income (k$)')

# Spending Score Distribution
plt.subplot(1, 3, 3)
sns.histplot(df['SpendingScore'], bins=20, color='green', kde=True)
plt.title('Spending Score Distribution')
plt.xlabel('Spending Score (1-100)')

plt.tight_layout()
plt.savefig('distributions.png')
plt.show()
print("✅ Distribution plots saved!")

# ── Step 5: Gender Analysis ───────────────────────────
plt.figure(figsize=(6, 4))
df['Gender'].value_counts().plot(kind='bar', color=['steelblue', 'coral'])
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('gender_distribution.png')
plt.show()
print("✅ Gender plot saved!")

# ── Step 6: Find Optimal Clusters (Elbow Method) ─────
X = df[['AnnualIncome', 'SpendingScore']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, 'bo-', linewidth=2, markersize=8)
plt.title('Elbow Method — Finding Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.xticks(range(1, 11))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('elbow_method.png')
plt.show()
print("✅ Elbow method plot saved!")

# ── Step 7: Apply K-Means with 5 Clusters ────────────
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

print("\nCluster Distribution:")
print(df['Cluster'].value_counts().sort_index())

# ── Step 8: Visualize Clusters ────────────────────────
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
cluster_names = ['Careful', 'Standard', 'Target', 'Careless', 'Sensible']

plt.figure(figsize=(10, 7))
for i in range(5):
    cluster_data = df[df['Cluster'] == i]
    plt.scatter(cluster_data['AnnualIncome'],
                cluster_data['SpendingScore'],
                c=colors[i], label=f'Cluster {i+1}: {cluster_names[i]}',
                s=100, alpha=0.8, edgecolors='white', linewidth=0.5)

plt.scatter(scaler.inverse_transform(kmeans.cluster_centers_)[:, 0],
            scaler.inverse_transform(kmeans.cluster_centers_)[:, 1],
            c='black', marker='*', s=300, label='Centroids', zorder=5)

plt.title('Customer Segmentation — K-Means Clustering', fontsize=14, fontweight='bold')
plt.xlabel('Annual Income (k$)', fontsize=12)
plt.ylabel('Spending Score (1-100)', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('customer_segments.png')
plt.show()
print("✅ Cluster visualization saved!")

# ── Step 9: Cluster Analysis & Insights ──────────────
print("\n" + "="*50)
print("CUSTOMER SEGMENT INSIGHTS")
print("="*50)
cluster_summary = df.groupby('Cluster')[['Age', 'AnnualIncome', 'SpendingScore']].mean().round(2)
cluster_summary.index = [f'Cluster {i+1}: {cluster_names[i]}' for i in range(5)]
print(cluster_summary)

print("\n✅ Customer Segmentation Analysis Complete!")
print("📊 4 charts saved in your project folder")