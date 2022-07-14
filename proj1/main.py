from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# The following lines adjust the granularity of reporting. 
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

print("Imported modules.")

# Import the dataset.
dataset = pd.read_csv(filepath_or_buffer="Air_Production_Dataset 2.csv")

# Print the first rows of the dataset.
dataset.head()

#q1
dataset.describe()

#q2
dataset.corr()

#################################################
dataset_mean = dataset.mean()
dataset_std = dataset.std()
dataset_norm = (dataset - dataset_mean)/dataset_std
#################################################
print("Dataset normalized.")

# Split the dataset into the training set (80%) and the test set (20%).
# Write your code between the lines (~ 1 line)
#################################################
train_set_norm, test_set_norm = train_test_split(dataset, train_size=0.8, test_size=0.2, random_state=100)
print(train_set_norm.head())
#################################################
print("Dataset split.")

# Create an empty list that will eventually hold all created feature columns.
feature_columns = []

# Represent Production_O2_NM3 as a floating-point value.
Oxygen = tf.feature_column.numeric_column("Production_O2_NM3")
feature_columns.append(Oxygen)

# Represent Plant_Air_NM3 as a floating-point value.
Air = tf.feature_column.numeric_column("Plant_Air_NM3")
feature_columns.append(Air)

# Complete this section later for *Question7*
# Write your code between the lines (~ 2 lines)
#################################################
# Represent Instrument_Air_NM3 as a floating-point value.
#instrument_air = tf.feature_column.numeric_column("Instrument_Air_NM3")
#feature_columns.append(instrument_air)

#################################################

# Convert the list of feature columns into a layer that will later be fed into
# the model. 
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

# Define the plotting function.

def plot_the_loss_curve(epochs, mse):
  """Plot a curve of loss vs. epoch."""

  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Mean Squared Error")

  plt.plot(epochs, mse, label="Loss")
  plt.legend()
  plt.ylim([mse.min()*0.95, mse.max() * 1.03])
  plt.show()  

print("Defined the plot_the_loss_curve function.")

# Define functions to create and train a linear regression model
def create_model(learning_rate, feature_layer):
  """Create and compile a linear regression model."""
  # Most simple tf.keras models are sequential.
  model = tf.keras.models.Sequential()

  # Add the layer containing the feature columns to the model.
  model.add(feature_layer)

  # Add one linear layer to the model to yield a simple linear regressor.
  model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))

  # Construct the layers into a model that TensorFlow can execute.
  model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=learning_rate),
                loss="mean_squared_error",
                metrics=[tf.keras.metrics.MeanSquaredError()])

  return model           


def train_model(model, dataset, epochs, batch_size, label_name):
  """Feed a dataset into the model in order to train it."""

  # Split the dataset into features and label.
  features = {name:np.array(value) for name, value in dataset.items()}
  label = np.array(features.pop(label_name))
  history = model.fit(x=features, y=label, batch_size=batch_size,
                      epochs=epochs, shuffle=True)
  
  # Gather the trained model's weight and bias.
  trained_weight = model.get_weights()[0]
  trained_bias = model.get_weights()[1]

  # Get details that will be useful for plotting the loss curve.
  epochs = history.epoch
  hist = pd.DataFrame(history.history)
  mse = hist["mean_squared_error"]

  return trained_weight, trained_bias, epochs, mse

print("Defined the create_model and train_model functions.")

# The following variables are the hyperparameters.
# Write your code between the lines (~ 3 lines)
#################################################
learning_rate = 0.1
epochs = 200
batch_size = 7
#################################################

label_name = "Energy_Input_MJ"

# Establish the model's topography.
model_1 = create_model(learning_rate, feature_layer)

# Train the model on the normalized training set.
weight, bias, epochs, mse = train_model(model_1, train_set_norm, epochs, batch_size, label_name)
plot_the_loss_curve(epochs, mse)

print("\nThe learned weight for your model is", weight)
print("The learned bias for your model is", bias )

# Evaluate the trained model against the test set.
test_features = {name:np.array(value) for name, value in test_set_norm.items()}
test_label = np.array(test_features.pop(label_name)) # isolate the label
print("\n Evaluate the trained linear regression model against the test set:")
model_1.evaluate(x = test_features, y = test_label, batch_size=7)