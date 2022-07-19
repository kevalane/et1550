import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns; 
sns.set()
from sklearn.decomposition import PCA

# The following lines adjust the granularity of reporting. 
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

print("Imported modules.")

# Import the dataset.
dataset = pd.read_csv(filepath_or_buffer="EU_Development_Indicators_Dataset.csv")

# Print some rows of the dataset.
dataset

# Get statistics on the dataset.
# Write your code between the lines (~ 1 line)
#################################################
dataset.describe()
#################################################

# Split the dataset into input features and the output target.
# Write your code between the lines (~ 2 lines)
#################################################
y_dataset = dataset[["Country_Name"]]
X_dataset = dataset.iloc[:, 1:]
#################################################
print("Dataset separated.")

# Convert raw values to their Z-scores
# Calculate the Z-scores of each input feature column.
# Write your code between the lines (~ 3 lines)
#################################################
X_dataset_mean = X_dataset.mean()
X_dataset_std = X_dataset.std()
X_dataset_norm =  (X_dataset - X_dataset_mean) / X_dataset_std
#################################################
print("Dataset normalized.")

# Set number of components to 2 and then fit the PCA model.
# Write your code between the lines (~ 2 lines)
#################################################
pca = PCA(n_components=2)
pca.fit(X_dataset_norm)
#################################################

print(pca.components_) #https://stackoverflow.com/questions/50796024/feature-variable-importance-after-a-pca-analysis

# Find the total ratio of the retained variance.
# Write your code between the lines (~ 1 line)
#################################################
total_explained_variance_ratio = np.sum(pca.explained_variance_ratio_) # thanks github copilot, have you tried it?
#################################################
print(total_explained_variance_ratio)

# Transform the normalized dataset.
# Write your code between the lines (~ 1 line)
#################################################
X_pca = pca.transform(X_dataset_norm)
#################################################
print("original shape:   ", X_dataset_norm.shape)
print("transformed shape:", X_pca.shape)

plt.figure(figsize=(11,11))
plt.xlabel('principal component 1')
plt.ylabel('principal component 2')
for i,type in enumerate(y_dataset):
    (x1, x2) = X_pca[i]
    plt.scatter(x1, x2)
    plt.text(x1+0.05, x2+0.05, type, fontsize=10)
plt.show()