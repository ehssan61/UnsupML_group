#!/usr/bin/env python
# coding: utf-8

# **Shill Bidding Dataset**
#
# **Description**
#
# Creators scraped a large number of eBay auctions of a popular product. After preprocessing the auction data, they created the SB dataset.   It is a multivariate dataset with 6321 instances and 13 features.
#
# **Location**
#
# https://archive.ics.uci.edu/dataset/562/shill+bidding+dataset
#
# **Variable Information**
#
# - Record ID: Unique identifier of a record in the dataset.
# - Auction ID: Unique identifier of an auction.
# - Bidder ID: Unique identifier of a bidder.
# - Bidder Tendency: A shill bidder participates exclusively in auctions of few sellers rather than a diversified lot.  This is a collusive act involving the fraudulent seller and an accomplice.
# - Bidding Ratio: A shill bidder participates more frequently to raise the auction price and attract higher bids from legitimate participants.
# - Successive Outbidding: A shill bidder successively outbids himself even though he is the current winner to increase the price gradually with small consecutive increments.
# - Last Bidding: A shill bidder becomes inactive at the last stage of the auction (more than 90\% of the auction duration) to avoid winning - the auction.
# - Auction Bids: Auctions with SB activities tend to have a much higher number of bids than the average of bids in concurrent auctions.
# - Auction Starting Price:  a shill bidder usually offers a small starting price to attract legitimate bidders into the auction.
# - Early Bidding: A shill bidder tends to bid pretty early in the auction (less than 25\% of the auction duration) to get the attention of auction users.
# - Winning Ratio: A shill bidder competes in many auctions but hardly wins any auctions.
# - Auction Duration:  How long an auction lasted.
# - Class: 0 for normal behaviour bidding; 1 for otherwise.

# In[1]:


import pandas as pd

# Download the dataset
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00562/Shill%20Bidding%20Dataset.csv"
url = "data/Shill Bidding Dataset.csv"  # use local dataset within the repo instead.

# Load into a DataFrame
df = pd.read_csv(url)

# Convert Record_ID to object
df["Record_ID"] = df["Record_ID"].astype(str)

# Convert Auction_ID to object
df["Auction_ID"] = df["Auction_ID"].astype(str)

# Set Record_ID as index
df.set_index("Record_ID", inplace=True)


# In[2]:


# Basic dataset info
df.info()

# First few rows
print(df.head())

# Shape of dataset (rows, columns)
print("Dataset shape:", df.shape)
# Check for duplicate records
print(f"Number of duplicate rows: {df.duplicated().sum()}")


# In[3]:


print(df.isnull().sum())  # Count of missing values per column
print(df.isna().sum().sum())  # Total missing values in dataset


# No missing values

# In[4]:


print(df.describe())  # Summary statistics for numerical columns
# Exclude non-feature columns for analysis
exclude_columns = ["Record_ID", "Auction_ID"]
df_numeric = df.drop(columns=exclude_columns, errors="ignore").select_dtypes(
    include=["number"]
)

# Analyze Auction_ID to determine if it should be categorical
print(df["Auction_ID"].value_counts())

# Convert Auction_ID to categorical if it has a reasonable number of unique values
if (
    df["Auction_ID"].nunique() < df.shape[0] * 0.05
):  # Example threshold: less than 5% of dataset size
    df["Auction_ID"] = df["Auction_ID"].astype("category")
    print("Auction_ID converted to categorical.")
else:
    print("Auction_ID left as is.")

# Outlier detection using IQR method
import numpy as np

Q1 = df_numeric.quantile(0.25)
Q3 = df_numeric.quantile(0.75)
IQR = Q3 - Q1
outliers = ((df_numeric < (Q1 - 1.5 * IQR)) | (df_numeric > (Q3 + 1.5 * IQR))).sum()
print("Outliers per column:\n", outliers)


# In[5]:


for col in df.select_dtypes(include=["object"]).columns:
    print(f"{col}: {df[col].nunique()} unique values")
    print(df[col].value_counts())


# In[6]:


# Check column data types
print(df.dtypes)

# Select only numeric columns
df_numeric = df.select_dtypes(include=["number"])

# Check if df_numeric is empty (i.e., if all columns were non-numeric)
if df_numeric.shape[1] == 0:
    print("No numeric columns found. Check your dataset.")
else:
    print("Numeric columns found:", df_numeric.columns)


# In[7]:


for col in df_numeric.columns:
    non_numeric_values = df[col][
        ~df[col].astype(str).str.replace(".", "", 1).str.isnumeric()
    ]
    if not non_numeric_values.empty:
        print(f"Non-numeric values found in column '{col}':")
        print(non_numeric_values.unique())


# In[8]:


df["Last_Bidding"] = pd.to_numeric(df["Last_Bidding"], errors="coerce")
df["Early_Bidding"] = pd.to_numeric(df["Early_Bidding"], errors="coerce")


# In[9]:


print(df[["Last_Bidding", "Early_Bidding"]].dtypes)  # Should now be float64
print(df[["Last_Bidding", "Early_Bidding"]].isna().sum())  # Check again for NaNs


# In[10]:


df_numeric = df.select_dtypes(include=["number"])  # Keep only numeric columns
df_numeric = df_numeric.drop(
    columns=["Bidder_ID"], errors="ignore"
)  # Drop Bidder_ID if present


# In[11]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()


# In[12]:


df_numeric = df.select_dtypes(include=["number"])  # Keep only numeric columns
df_numeric.hist(figsize=(12, 10), bins=20, edgecolor="black")
plt.show()


# In[13]:


import seaborn as sns

df_numeric = df.select_dtypes(include=["number"])  # Select numeric columns
sns.pairplot(df_numeric.sample(100))  # Sample to speed up plotting
plt.show()


# In[14]:


plt.figure(figsize=(15, 8))
df_numeric.boxplot(rot=90)
plt.title("Box Plots of Numeric Features")
plt.show()


# In[15]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

# Standardize before PCA
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_numeric)

# PCA transformation
pca = PCA(n_components=None)
df_pca = pca.fit_transform(df_scaled)

# Cumulative variance plot
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker="o")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Explained Variance by PCA Components")
plt.grid()
plt.show()

# If PC1 and PC2 explain less than 80% variance, consider a 3D PCA plot
if np.cumsum(pca.explained_variance_ratio_)[1] < 0.80:
    from mpl_toolkits.mplot3d import Axes3D

    pca_3d = PCA(n_components=3)
    df_pca_3d = pca_3d.fit_transform(df_scaled)

    # Convert to DataFrame
    df_pca_3d = pd.DataFrame(df_pca_3d, columns=["PC1", "PC2", "PC3"])

    # 3D Scatter plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(df_pca_3d["PC1"], df_pca_3d["PC2"], df_pca_3d["PC3"], alpha=0.5)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.title("3D PCA Visualization")
    plt.show()


# In[ ]:


# In[ ]:
