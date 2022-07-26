from __future__ import absolute_import, division, print_function, unicode_literals

#Import relevant modules
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# The following lines adjust the granularity of reporting. 
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

print("Imported modules.")

df = load_breast_cancer()
dataset= pd.DataFrame(df['data'], columns=df['feature_names'])
dataset['target']= df['target']
dataset.columns = dataset.columns.str.replace(' ', '_')

dataset.head()
dataset.describe()

# Generate a correlation matrix.
# Write your code between the lines (~ 1 line)
#################################################
# dataset.corr() # for entire corr() matrix
print(dataset.corr()["target"].to_string()) # smoothness_error 0.1
#################################################

# Split the dataset into input features and the output target.
# Write your code between the lines (~ 2 line)
#################################################
y_dataset = dataset[["target"]]
X_dataset = dataset.iloc[:, :-1]
#################################################
print("Dataset separated.")

# Convert raw values to their Z-scores
# Calculate the Z-scores of each input feature column.
# Write your code between the lines (~ 3 lines)
#################################################
X_dataset_mean = X_dataset.mean()
X_dataset_std = X_dataset.std()
X_dataset_norm = (X_dataset - X_dataset_mean)/X_dataset_std
#################################################
print("Dataset normalized.")

# Split the dataset into the training set (90%) and the test set (10%).
# Write your code between the lines (~ 1 line)
#################################################
X_train_norm, X_test_norm, y_train, y_test = train_test_split(X_dataset_norm, y_dataset, train_size=0.9, test_size=0.1, random_state=110)
#################################################
print("Dataset split.")

# Define the plotting function
def plot_curve(epochs, hist, list_of_metrics):
  """Plot a curve of one or more metrics vs. epoch."""  
  

  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Value")

  for m in list_of_metrics:
    x = hist[m]
    plt.plot(epochs[1:], x[1:], label=m)

  plt.legend()
  plt.show()

print("Loaded the plot_curve function.")

# Define functions to create and train a logistic regression model
def create_LR_model(learning_rate):
  """Create and compile a logistic regression model."""
  # Most simple tf.keras models are sequential.
  model = tf.keras.models.Sequential()

  # Add a layer to the model to yield a sigmoid function.
  # Write your code between the lines (~ 1 line)
  #################################################
  model.add(tf.keras.layers.Dense(units=1 , activation='sigmoid'))
  #################################################

  # Construct the layers into a model that TensorFlow can execute.
  model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                loss="binary_crossentropy",
                metrics=[tf.keras.metrics.BinaryAccuracy()])

  return model           


def train_LR_model(model, train_features, train_label, epochs, batch_size=None):
  """Feed a dataset into the model in order to train it."""
  
  history = model.fit(x=train_features, y=train_label, batch_size=batch_size,
                      epochs=epochs, shuffle=False)

  # Get details that will be useful for plotting the loss curve.
  epochs = history.epoch
  hist = pd.DataFrame(history.history)

  return epochs, hist

print("Defined the create_model and train_model functions.")

# The following variables are the hyperparameters.
learning_rate = 0.01
epochs = 100
batch_size = 200

# Establish the model's topography.
model_1 = create_LR_model(learning_rate)

# Train the model on the normalized training set.
epochs, hist = train_LR_model(model_1, X_train_norm, y_train, epochs, batch_size)

# Plot a graph of the metric vs. epochs.
list_of_metrics_to_plot = ['binary_accuracy']
plot_curve(epochs, hist, list_of_metrics_to_plot)

# Evaluate the trained model against the test set.
print("\n Evaluate the logistic regression model against the test set:")
model_1.evaluate(x = X_test_norm, y = y_test)

def create_NN_model(learning_rate):
  """Create and compile a neural network model."""
  # Most simple tf.keras models are sequential.
  model = tf.keras.models.Sequential()

  # Define the first hidden layer with 16 nodes.   
  model.add(tf.keras.layers.Dense(units=16, 
                                  activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.l2(l=0.001),
                                  name='Hidden1'))
  
  # Write your code between the lines (~ few lines)
  #################################################  
  # Define the second hidden layer with 8 nodes.   
  model.add(tf.keras.layers.Dense(units=8, 
                                  activation='relu', 
                                  kernel_regularizer=tf.keras.regularizers.l2(l=0.001), 
                                  name='Hidden2'))
  
  
  # Define the third hidden layer with 6 nodes.
  model.add(tf.keras.layers.Dense(units=6, 
                                  activation='relu', 
                                  kernel_regularizer=tf.keras.regularizers.l2(l=0.001), 
                                  name='Hidden3'))

  
  #################################################
  
  # Define the output layer.
  model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))                              
  
  model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                loss="binary_crossentropy",
                metrics=[tf.keras.metrics.BinaryAccuracy()])

  return model

def train_NN_model(model, train_features, train_label, epochs, batch_size=None):
    """Feed a dataset into the model in order to train it."""

    history = model.fit(x=train_features, y=train_label, batch_size=batch_size,
                      epochs=epochs, shuffle=False)

    # Get details that will be useful for plotting the loss curve.
    epochs = history.epoch
    hist = pd.DataFrame(history.history)

    return epochs, hist

print("Defined the create_model and train_model functions.")

# The following variables are the hyperparameters.
learning_rate = 0.01
epochs = 100
batch_size = 200

# Establish the model's topography.
model_2 = create_NN_model(learning_rate)

# Train the model on the normalized training set.
epochs, hist = train_NN_model(model_2, X_train_norm, y_train, epochs, batch_size)

# Plot a graph of the metric vs. epochs.
list_of_metrics_to_plot = ['binary_accuracy']
plot_curve(epochs, hist, list_of_metrics_to_plot)

# Evaluate the trained model against the test set.
print("\n Evaluate the neural network model against the test set:")
model_2.evaluate(x = X_test_norm, y = y_test)