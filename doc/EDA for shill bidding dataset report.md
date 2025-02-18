**Exploratory Data Analysis (EDA) for Shill Bidding Dataset**

The goal of this EDA is to prepare the dataset for unsupervised clustering to detect shill bidder fraud. The analysis provides data integrity, identifies potential transformations, and identifies  MinMax Scaling as a necessary process prior to clustering.

**1. Data Cleaning & Preprocessing**

- Record_ID was converted to an object and set as the index, as it uniquely identifies each row and should not be included in clustering analysis.
- Auction_ID was analyzed to determine whether it should be treated as a categorical variable:
  - If Auction_ID had a reasonable number of unique values (less than 5% of dataset size), it was going to be converted to categorical, but it didn’t.  
- Verified there were no duplicate records that could bias clustering outcomes.
- Assessed missing values and confirmed that no substantial gaps required imputation.

**2. Exploratory Data Analysis**

Summary Statistics & Data Distribution
df.describe() was used to inspect key statistics (mean, min, max, standard deviation) for all numerical variables.
A correlation heatmap was generated to identify redundant features that might not contribute useful information to clustering.

[create link to Correlation Heatmap in fig folder]

Histograms provided insight into feature distributions and possible skewness.

[create link to Histograms in fig folder]

Boxplots were used to identify extreme outliers that might impact clustering results.

[create link to Box Plot image in fig folder]

Interquartile Range (IQR) Analysis quantified the number of extreme values in each feature, but no action was taken to remove outliers as they might be relevant for fraud detection.

3. Dimensionality Reduction Using PCA

Principal Component Analysis (PCA) was applied to check whether dimensionality reduction would be necessary.
  - A cumulative variance plot was generated to determine how many components explain sufficient variance:
  - If the first two components explained more than 80% of the variance, a 2D PCA plot would suffice.
  - A 3D PCA visualization was also generated to analyze data structure in three principal components.

[create link to PCA figure]


4. MinMax Scaling

- Clustering algorithms like K-Means, DBSCAN, and Spectral Clustering use distance-based calculations, which can be biased if features have different scales.
- MinMax Scaling ensures each feature contributes equally to clustering by normalizing all values into a [0,1] range.
- StandardScaler standardizes data to zero mean and unit variance, which is ideal for PCA but unnecessary for clustering.
- MinMax Scaling preserves the original distribution and relationships between values, making it a better choice for clustering.


Note:
Additional feature engineering may be required if clustering does not yield meaningful group separations.
