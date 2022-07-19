import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns; 
sns.set()
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from scipy.stats import mode
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

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
#for i,type in enumerate(y_dataset):
for i, type in np.ndenumerate(y_dataset):
    (x1, x2) = X_pca[i[0]]
    plt.scatter(x1, x2)
    plt.text(x1+0.05, x2+0.05, type, fontsize=10)
plt.show()


#################################################
# PART 2:
#################################################


digits = load_digits()
X_dataset_ = digits.data
y_dataset_ = digits.target

print('Dataset loaded.')

# Define the k-means clustering model and fit the model.
# Write your code between the lines (~ 2 lines)
#################################################
kmeans = KMeans(n_clusters=10, random_state=0)
kmeans.fit(X_dataset_)
#################################################

# Find cluster index for each data point.
# Write your code between the lines (~ 1 line)
#################################################
clusters = kmeans.fit_predict(X_dataset_)
#################################################

kmeans.cluster_centers_.shape

fig, ax = plt.subplots(2, 5, figsize=(8, 3))
centers = kmeans.cluster_centers_.reshape(10, 8, 8)
for axi, center in zip(ax.flat, centers):
    axi.set(xticks=[], yticks=[])
    axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)

labels = np.zeros_like(clusters)
for i in range(10):
    mask = (clusters == i)
    labels[mask] = mode(y_dataset_[mask])[0]

print('Cluster indexes fixed.')

# Find the accuracy score of the k-means clustering.
# Write your code between the lines (~ 1 line)
#################################################
kmeans_accuracy = accuracy_score(y_dataset_, labels)
#################################################
print(kmeans_accuracy)

mat = confusion_matrix(digits.target, labels)
plt.figure(figsize=(8,8))
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=digits.target_names,
            yticklabels=digits.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')