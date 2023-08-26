#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

# read in the data
data = pd.read_csv(r"C:\Users\Phoebe Tan\OneDrive - National University of Singapore\Desktop\Taiwan\Data Mining\HW3\training.csv", sep=',', header=0, skipinitialspace=True)
#data.columns = [col.strip() for col in data.columns]


# In[2]:


data


# # Data Exploration

# In[3]:


#Data Exploration
# libraries
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import scale


# In[4]:


# about the dataset

# dimensions
print("Dimensions: ", data.shape, "\n")

# data types
print(data.info())

# head
data.head()


# In[5]:


print(data.columns)


# In[6]:


order = list(np.sort(data['lettr'].unique()))
print(order)


# In[7]:


# basic plots: How do various attributes vary with the letters

plt.figure(figsize=(16, 8))
sns.barplot(x='lettr', y='x-box', 
            data=data, 
            order=order)


# In[8]:


letter_means = data.groupby('lettr').mean()
letter_means.head()


# In[9]:


plt.figure(figsize=(18, 10))
sns.heatmap(letter_means)


# # Data Preprocessing

# In[10]:


# average feature values
round(data.drop('lettr', axis=1).mean(), 2)


# In[11]:


#Relabel Data

# Create a new column "label" with binary labels (1 for normal, 0 for abnormal)
data['label'] = data['lettr'].apply(lambda x: 1 if x in ['B', 'H', 'P', 'W', 'R', 'M'] else 0)


# In[12]:


# splitting into X and y
X_train = data.drop(["lettr","label"], axis = 1)
y_train = data['label']


# In[13]:


# scaling the features
X_train = scale(X_train)
unscaled=data.drop(["lettr","label"], axis = 1)


# # Creating a Validation Set

# In[14]:


import pandas as pd
import numpy as np

# URL of the dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data'

# Column names for the dataset
column_names = ['lettr', 'x-box', 'y-box', 'width', 'high', 'onpix', 'x-bar', 'y-bar', 'x2bar', 'y2bar', 'xybar', 'x2ybr', 'xy2br', 'x-ege', 'xegvy', 'y-ege', 'yegvx']

# Read the dataset into a pandas DataFrame
data = pd.read_csv(url, names=column_names)

# Select the letters B, H, P, W, R, and M
selected_letters = ['B', 'H', 'P', 'W', 'R', 'M']
selected_data = data[data['lettr'].isin(selected_letters)]

# Randomly sample 600 instances from the selected letters
sampled_data = selected_data.sample(n=600, random_state=42)

# Randomly select 400 instances from other letters
other_letters_data = data[~data['lettr'].isin(selected_letters)]
other_sampled_data = other_letters_data.sample(n=400, random_state=42)

# Combine the sampled data
valid = pd.concat([sampled_data, other_sampled_data])

# Shuffle the validation data
valid = valid.sample(frac=1, random_state=42)

# Display the validation set data
print(valid)


# In[15]:


# Count the occurrences of each letter in the final data
letter_counts = valid['lettr'].value_counts()

# Calculate the proportion of normal and abnormal letters
normal_proportion = letter_counts.loc[['B', 'H', 'P', 'W', 'R', 'M']].sum() / len(valid)
abnormal_proportion = 1 - normal_proportion

# Display the proportions
print("Proportion of Normal Letters:", normal_proportion)
print("Proportion of Abnormal Letters:", abnormal_proportion)


# In[16]:


#Relabel Data

# Create a new column "label" with binary labels (1 for normal, 0 for abnormal)
valid['label'] = valid['lettr'].apply(lambda x: 1 if x in ['B', 'H', 'P', 'W', 'R', 'M'] else 0)


# In[17]:


# splitting into X and y
X_valid = valid.drop(["lettr","label"], axis = 1)
y_valid = valid['label']


# In[18]:


# scaling the features
X_valid = scale(X_valid)


# In[19]:


print(type(y_train))


# In[20]:



# Load the test data
test_data = pd.read_csv('C:/Users/Phoebe Tan/OneDrive - National University of Singapore/Desktop/Taiwan/Data Mining/HW3/test_X.csv')

# Prepare the test data
X_test = test_data.values  # Assuming the test data is in a DataFrame

# Standardize the test data using the same scaler used during training
X_test_scaled = scale(X_test)


# # Method One: One-Class SVM
# 

# 
# import numpy as np
# from sklearn.svm import OneClassSVM
# from sklearn.model_selection import GridSearchCV
# 

# In[21]:


from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, ParameterGrid

# Define the parameter grid for hyperparameters
param_grid = {
    'gamma': [0.05,0.075,0.1,0.12,0.15,0.17,0.19,0.2],
    'nu': [0.05, 0.1, 0.5, 0.9,0.075,0.085,0.04,0.3,0.5],
    'kernel': ['sigmoid', 'poly','rbf','linear']
}

best_auc_score = 0.0
best_params = None

# Generate all parameter combinations
parameter_combinations = list(ParameterGrid(param_grid))

# Perform grid search and select best hyperparameters based on AUC score
for params in parameter_combinations:
    # Create a new model with the current parameter combination
    model = OneClassSVM(**params)
    
    # Train the model on X_train
    model.fit(X_train)
    
    # Make predictions on X_valid
    predictions = model.predict(X_valid)
    
    # Calculate the AUC score
    auc_score = roc_auc_score(y_valid, predictions)
    
    # Check if this parameter combination has a better AUC score
    if auc_score > best_auc_score:
        best_auc_score = auc_score
        best_params = params

# Train the final model with the best parameters using the entire X_train dataset
final_model = OneClassSVM(**best_params)
model=final_model.fit(X_train)


# In[56]:


from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, ParameterGrid

param_grid = {
    'gamma': [0.001, 0.01, 0.1,0.095,0.12,0.15],
    'nu': [0.9,0.95,0.1, 0.12,0.15,0.2, 0.3],
    'kernel': ['rbf', 'sigmoid']
}

best_auc_score = 0.0
best_params = None

# Generate all parameter combinations
parameter_combinations = list(ParameterGrid(param_grid))

# Perform grid search and select best hyperparameters based on AUC score
for params in parameter_combinations:
    # Create a new model with the current parameter combination
    model = OneClassSVM(**params)
    
    # Train the model on X_train
    model.fit(X_train)
    
    # Make predictions on X_valid
    predictions = model.predict(X_valid)
    
    # Calculate the AUC score
    auc_score = roc_auc_score(y_valid, predictions)
    
    # Check if this parameter combination has a better AUC score
    if auc_score > best_auc_score:
        best_auc_score = auc_score
        best_params = params

# Train the final model with the best parameters using the entire X_train dataset
final_model = OneClassSVM(**best_params)
final_model.fit(X_train)


# In[26]:


from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, ParameterGrid

# Define the parameter grid for hyperparameters
param_grid = {
    'gamma': [0.15,0.17,0.19,0.2,0.23,0.25,0.17,0.1],
    'nu': [0.05, 0.1, 0.5, 0.9,0.075,0.085,0.04,0.3,0.5,0.045,0.37],
    'kernel': ['sigmoid', 'poly','rbf','linear']
}

best_auc_score = 0.0
best_params = None

# Generate all parameter combinations
parameter_combinations = list(ParameterGrid(param_grid))

# Perform grid search and select best hyperparameters based on AUC score
for params in parameter_combinations:
    # Create a new model with the current parameter combination
    model = OneClassSVM(**params)
    
    # Train the model on X_train
    model.fit(X_train)
    
    # Make predictions on X_valid
    predictions = model.predict(X_valid)
    
    # Calculate the AUC score
    auc_score = roc_auc_score(y_valid, predictions)
    
    # Check if this parameter combination has a better AUC score
    if auc_score > best_auc_score:
        best_auc_score = auc_score
        best_params = params

# Train the final model with the best parameters using the entire X_train dataset
final_model = OneClassSVM(**best_params)
model=final_model.fit(X_train)


# In[30]:


from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, ParameterGrid

# Define the parameter grid for hyperparameters
param_grid = {
    'gamma': ['auto'],
    'nu': [0.05],
    'kernel': ['rbf']
}

best_auc_score = 0.0
best_params = None

# Generate all parameter combinations
parameter_combinations = list(ParameterGrid(param_grid))

# Perform grid search and select best hyperparameters based on AUC score
for params in parameter_combinations:
    # Create a new model with the current parameter combination
    model = OneClassSVM(**params)
    
    # Train the model on X_train
    model.fit(X_train)
    
    # Make predictions on X_valid
    predictions = model.predict(X_valid)
    
    # Calculate the AUC score
    auc_score = roc_auc_score(y_valid, predictions)
    
    # Check if this parameter combination has a better AUC score
    if auc_score > best_auc_score:
        best_auc_score = auc_score
        best_params = params

# Train the final model with the best parameters using the entire X_train dataset
final_model = OneClassSVM(**best_params)
model=final_model.fit(unscaled)


# In[37]:



# Train the final model with the best parameters using the entire X_train dataset
final_model = OneClassSVM(**best_params)
model=final_model.fit(unscaled)


# In[38]:


best_params #Fifth Time Best Time


# In[27]:


best_params #Fourth Time Best Time


# In[23]:


best_params #Third Time 


# In[49]:


best_params #Second time


# In[54]:


best_params #First time


# In[27]:


best_auc_score


# In[44]:


# Extract anomaly scores from the model
anomaly_scores = final_model.decision_function(X_valid)
anomaly_scores= -anomaly_scores


# In[39]:


from sklearn.metrics import classification_report


# Extract anomaly scores from the model
anomaly_scores = model.decision_function(X_test)
anomaly_scores= -anomaly_scores
anomaly_scores


# In[40]:


import pandas as pd

# Assuming 'anomaly_scores' is a list or array containing the anomaly scores

# Create a DataFrame to hold the anomaly scores
scores_df = pd.DataFrame({'Anomaly Score': anomaly_scores})

# Save the DataFrame to a CSV file
scores_df.to_csv('anomaly_scores95.csv', index=False)


# # Method 2: KNN

# In[21]:


import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, ParameterGrid

# Determine the number of clusters (n)
n_clusters = 5  # Example value, adjust as per your dataset

# Perform K-means clustering on the training data
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(X_train)

# Get the centroids of the clusters
centroids = kmeans.cluster_centers_

# Calculate the distances between the training data points and the centroids
distances = kmeans.transform(X_train)

# Calculate the weights based on the distances
weights = 1 / distances

# Transpose the weights array for dot product
weights = weights.T

# Define the parameter grid for hyperparameters
param_grid = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
}

best_auc_score = 0.0
best_params = None

# Generate all parameter combinations
parameter_combinations = list(ParameterGrid(param_grid))

# Perform grid search and select the best hyperparameters based on AUC score
for params in parameter_combinations:
    # Create a new model with the current parameter combination
    model = KNeighborsClassifier(**params)
    
    # Train the model on X_train with adjusted weights
    weighted_y_train = np.dot(weights, y_train)
    model.fit(X_train, weighted_y_train)
    
    # Make predictions on X_valid
    predictions = model.predict(X_valid)
    
    # Calculate the AUC score
    auc_score = roc_auc_score(y_valid, predictions)
    
    # Check if this parameter combination has a better AUC score
    if auc_score > best_auc_score:
        best_auc_score = auc_score
        best_params = params

# Train the final model with the best parameters using the entire X_train dataset
final_model = KNeighborsClassifier(**best_params)
final_model.fit(X_train, y_train)


# In[26]:


import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

# Step 1: Calculate centroids using K-means on the training data

def calculate_centroids(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    centroids = kmeans.cluster_centers_
    return centroids

# Step 2: Cluster the testing data based on the centroids

def cluster_data(X, centroids):
    clusters = []
    for x in X:
        distances = [np.linalg.norm(x - centroid) for centroid in centroids]
        cluster = np.argmin(distances)
        clusters.append(cluster)
    return clusters

# Step 3: Train the model and evaluate using AUC score

def train_and_evaluate(X_train, y_train, X_valid, y_valid, n_clusters):
    # Calculate centroids
    centroids = calculate_centroids(X_train, n_clusters)

    # Cluster the validation data
    valid_clusters = cluster_data(X_valid, centroids)

    # Calculate distances for each data point
    distances = [np.linalg.norm(X_valid[i] - centroids[valid_clusters[i]]) for i in range(len(X_valid))]

    # Calculate AUC score
    auc_score = roc_auc_score(y_valid, distances)

    return auc_score


n_clusters = 2 # Set the desired number of clusters

auc_score = train_and_evaluate(X_train, y_train, X_valid, y_valid, n_clusters)
print("AUC Score:", auc_score)


# # Method 3: Auto Encoders

# In[21]:


import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Define and train the AE on the training data
def train_autoencoder(X_train):
    input_dim = X_train.shape[1]

    # Define the AE model architecture
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(11, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    autoencoder = Model(input_layer, decoded)

    # Compile and train the model
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(X_train, X_train, epochs=10, batch_size=32)

    # Return the trained model
    return autoencoder


# Train the AE on the training data
autoencoder = train_autoencoder(X_train)

# Use the trained AE to reconstruct the validation data
reconstructed_valid = autoencoder.predict(X_valid)

# Calculate the reconstruction loss for each validation data point
mse = np.mean(np.square(X_valid - reconstructed_valid), axis=1)

# Calculate the AUC score using the reconstruction loss as weight values
auc_score = roc_auc_score(y_valid, mse)
print("AUC Score:", auc_score)


# In[21]:


import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import regularizers
from sklearn.preprocessing import MinMaxScaler

# Normalize the input data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)

# Define and train the AE on the training data
def train_autoencoder(X_train):
    input_dim = X_train.shape[1]

    # Define the AE model architecture
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(128, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    autoencoder = Model(input_layer, decoded)

    # Add regularization to the encoded layer
    encoded = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)

    # Compile and train the model
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    autoencoder.fit(X_train, X_train, epochs=50, batch_size=32)

    # Return the trained model
    return autoencoder

# Train the AE on the training data
autoencoder = train_autoencoder(X_train)

# Use the trained AE to reconstruct the validation data
reconstructed_valid = autoencoder.predict(X_valid)

# Calculate the reconstruction loss for each validation data point
mse = np.mean(np.square(X_valid - reconstructed_valid), axis=1)

# Calculate the AUC score using the reconstruction loss as weight values
auc_score = roc_auc_score(y_valid, mse)
print("AUC Score:", auc_score)


# In[28]:


import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping


# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

# Define the Autoencoder architecture
input_dim = X_train.shape[1]  # Number of features
encoding_dim = 32  # Dimension of the encoded representation

input_layer = Input(shape=(input_dim,))
encoder_layer = Dense(encoding_dim, activation='relu')(input_layer)
decoder_layer = Dense(input_dim, activation='relu')(encoder_layer)
autoencoder = Model(input_layer, decoder_layer)

# Compile and train the Autoencoder
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
autoencoder.fit(X_train_scaled, X_train_scaled, epochs=50, batch_size=32, validation_data=(X_valid_scaled, X_valid_scaled),
                callbacks=[early_stopping])

# Use the trained Autoencoder for anomaly detection
X_valid_pred = autoencoder.predict(X_valid_scaled)
mse = np.mean(np.power(X_valid_scaled - X_valid_pred, 2), axis=1)

# Assign weight values based on reconstruction loss (MSE)
threshold = np.percentile(mse, 95)  # Adjust the percentile threshold as needed
weights = np.where(mse > threshold, mse, 1.0)

# Calculate AUC score for evaluation
auc_score = roc_auc_score(y_valid, weights)

print("AUC Score:", auc_score)


# In[29]:


weights


# In[31]:


import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping


# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

# Define the Autoencoder architecture
input_dim = X_train.shape[1]  # Number of features
encoding_dim = 8  # Dimension of the encoded representation

input_layer = Input(shape=(input_dim,))
encoder_layer = Dense(encoding_dim, activation='relu')(input_layer)
decoder_layer = Dense(input_dim, activation='relu')(encoder_layer)
autoencoder = Model(input_layer, decoder_layer)

# Compile and train the Autoencoder
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
autoencoder.fit(X_train_scaled, X_train_scaled, epochs=50, batch_size=32, validation_data=(X_valid_scaled, X_valid_scaled),
                callbacks=[early_stopping])

# Use the trained Autoencoder for anomaly detection
X_valid_pred = autoencoder.predict(X_valid_scaled)
mse = np.mean(np.power(X_valid_scaled - X_valid_pred, 2), axis=1)

# Calculate AUC score for evaluation
auc_score = roc_auc_score(y_valid, mse)

print("AUC Score:", auc_score)


# In[32]:


weights


# In[27]:


import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

# Define the Autoencoder architecture
input_dim = X_train.shape[1]  # Number of features
encoding_dim = 11  # Dimension of the encoded representation
D
input_layer = Input(shape=(input_dim,))
encoder_layer = Dense(encoding_dim, activation='relu')(input_layer)
decoder_layer = Dense(input_dim, activation='relu')(encoder_layer)
autoencoder = Model(input_layer, decoder_layer)

# Compile and train the Autoencoder
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
autoencoder.fit(X_train_scaled, X_train_scaled, epochs=50, batch_size=32, validation_data=(X_valid_scaled, X_valid_scaled),
                callbacks=[early_stopping])

# Use the trained Autoencoder for anomaly detection
X_valid_pred = autoencoder.predict(X_valid_scaled)
mse = np.mean(np.power(X_valid_scaled - X_valid_pred, 2), axis=1)

# Assign weight values based on reconstruction loss (MSE)
weights = mse  # Assigning MSE values directly as weights

# Calculate AUC score for evaluation
auc_score = roc_auc_score(y_valid, weights)

print("AUC Score:", auc_score)


# In[67]:





# Use the trained Autoencoder for anomaly detection
X_test_pred = autoencoder.predict(X_test_scaled)
mse = np.mean(np.power(X_test_scaled - X_test_pred, 2), axis=1)

# Assign weight values based on reconstruction loss (MSE)
weights = mse  # Assuming you want to use the MSE as weights

# Add the weights as a new column in the test data DataFrame
test_data['Weights'] = weights

# Export the test data with weights to a CSV file
output_path = 'C:/Users/Phoebe Tan/OneDrive - National University of Singapore/Desktop/Taiwan/Data Mining/HW3/test_data_with_weights5.csv'
test_data.to_csv(output_path, index=False)


# In[61]:


weights


# In[22]:


import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tensorflow.keras import layers, losses, callbacks
from sklearn.model_selection import ParameterGrid

# Define the Autoencoder model
def create_autoencoder(input_shape, latent_dim):
    inputs = tf.keras.Input(shape=input_shape)
    encoded = layers.Dense(latent_dim, activation='relu')(inputs)
    decoded = layers.Dense(input_shape, activation='sigmoid')(encoded)
    autoencoder = tf.keras.Model(inputs, decoded)
    return autoencoder

# Set hyperparameters and initialize the model
input_shape = X_train.shape[1]  # Input shape of the images
epochs = 50  # Number of training epochs
batch_size = 32  # Batch size for training

# Define the grid of hyperparameters to search
param_grid = {
    'latent_dim': [5,10,15],
    'learning_rate': [0.001, 0.01, 0.1]
}

best_roc_auc = 0.0
best_params = {}

# Iterate over the parameter grid
for params in ParameterGrid(param_grid):
    print("Training with hyperparameters:", params)
    
    autoencoder = create_autoencoder(input_shape, params['latent_dim'])
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
                        loss=losses.MeanSquaredError())

    # Train the autoencoder
    history = autoencoder.fit(X_train, X_train,  # Use X_train as both input and target
                              epochs=epochs,
                              batch_size=batch_size,
                              validation_data=(X_valid, X_valid),
                              callbacks=[callbacks.EarlyStopping(patience=3)])
    
    # Calculate reconstruction error on validation set
    X_valid_pred = autoencoder.predict(X_valid)
    reconstruction_error = np.mean(np.square(X_valid_pred - X_valid), axis=1)

    # Calculate ROC AUC score
    roc_auc = roc_auc_score(y_valid, reconstruction_error)
    print("Validation ROC AUC Score:", roc_auc)
    
    # Check if the current model is the best
    if roc_auc > best_roc_auc:
        best_roc_auc = roc_auc
        best_params = params
        best_autoencoder = autoencoder



# In[23]:


# Print the best parameters and ROC AUC score
print("Best Parameters:", best_params)
print("Best ROC AUC Score:", best_roc_auc)

# Use the best model to predict with X_test
X_test_pred = best_autoencoder.predict(X_test_scaled)
reconstruction_error_test = np.mean(np.square(X_test_pred - X_test_scaled), axis=1)

# Use the reconstruction error as the weight value for prediction
predictions = reconstruction_error_test

# Print the predictions
print("Predictions:", predictions)


# In[24]:


import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tensorflow.keras import layers, losses, callbacks
from sklearn.model_selection import ParameterGrid

# Define the Autoencoder model
def create_autoencoder(input_shape, latent_dim):
    inputs = tf.keras.Input(shape=input_shape)
    encoded = layers.Dense(latent_dim, activation='relu')(inputs)
    decoded = layers.Dense(input_shape, activation='sigmoid')(encoded)
    autoencoder = tf.keras.Model(inputs, decoded)
    return autoencoder

# Set hyperparameters and initialize the model
input_shape = X_train.shape[1]  # Input shape of the images
epochs = 50  # Number of training epochs
batch_size = 32  # Batch size for training

# Define the grid of hyperparameters to search
param_grid = {
    'latent_dim': [10,11,12,13,14,15],
    'learning_rate': [0.001,0.01,0.005]
}

best_roc_auc = 0.0
best_params = {}

# Iterate over the parameter grid
for params in ParameterGrid(param_grid):
    print("Training with hyperparameters:", params)
    
    autoencoder = create_autoencoder(input_shape, params['latent_dim'])
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
                        loss=losses.MeanSquaredError())

    # Train the autoencoder
    history = autoencoder.fit(X_train, X_train,  # Use X_train as both input and target
                              epochs=epochs,
                              batch_size=batch_size,
                              validation_data=(X_valid, X_valid),
                              callbacks=[callbacks.EarlyStopping(patience=3)])
    
    # Calculate reconstruction error on validation set
    X_valid_pred = autoencoder.predict(X_valid)
    reconstruction_error = np.mean(np.square(X_valid_pred - X_valid), axis=1)

    # Calculate ROC AUC score
    roc_auc = roc_auc_score(y_valid, reconstruction_error)
    print("Validation ROC AUC Score:", roc_auc)
    
    # Check if the current model is the best
    if roc_auc > best_roc_auc:
        best_roc_auc = roc_auc
        best_params = params
        best_autoencoder = autoencoder



# In[26]:


import pandas as pd

# Print the best parameters and ROC AUC score
print("Best Parameters:", best_params)
print("Best ROC AUC Score:", best_roc_auc)

# Use the best model to predict with X_test
X_test_pred = best_autoencoder.predict(X_test_scaled)
reconstruction_error_test = np.mean(np.square(X_test_pred - X_test_scaled), axis=1)

# Use the reconstruction error as the weight value for prediction
predictions = reconstruction_error_test

# Create a DataFrame with predictions
df_predictions = pd.DataFrame({'Predictions': predictions})

# Export the DataFrame to a CSV file
df_predictions.to_csv('predictions.csv', index=False)


# # Isolation Forest

# In[30]:


import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_samples': [0.1, 0.2, 0.5],
    'contamination': [0.01, 0.05, 0.1]
    # Add more hyperparameters as needed
}

best_auc = 0.0
best_model = None

# Iterate over all parameter combinations
for n_estimators in param_grid['n_estimators']:
    for max_samples in param_grid['max_samples']:
        for contamination in param_grid['contamination']:
            # Train the Isolation Forest model with the current hyperparameters
            model = IsolationForest(n_estimators=n_estimators, max_samples=max_samples, contamination=contamination)
            model.fit(X_train)

            # Predict anomaly scores for the validation set
            valid_scores = model.decision_function(X_valid)

            # Calculate ROC AUC score on the validation set
            auc_score = roc_auc_score(y_valid, valid_scores)

            # Check if the current model is the best one so far
            if auc_score > best_auc:
                best_auc = auc_score
                best_model = model

# Predict anomaly scores for the test set using the best model
test_scores = best_model.decision_function(X_test_scaled)

# Output the anomaly scores for X_test
print("Anomaly Scores for X_test:")
print(test_scores)


# In[31]:


# Use the reconstruction error as the weight value for prediction
predictions = test_scores

# Create a DataFrame with predictions
df_predictions = pd.DataFrame({'outliers': predictions})

# Export the DataFrame to a CSV file
df_predictions.to_csv('predictions3.csv', index=False)


# In[32]:


best_auc


# In[33]:


import numpy as np
from sklearn.ensemble import IsolationForest

# Train the Isolation Forest model
isolation_forest = IsolationForest()
isolation_forest.fit(X_train)

# Predict anomaly scores for the training set
train_scores = isolation_forest.decision_function(X_train)

# Assign weight values based on anomaly scores for the training set
train_weights = np.where(train_scores < 0, 1, 0)

# Predict anomaly scores for the test set
test_scores = isolation_forest.decision_function(X_test_scaled)

# Assign weight values based on anomaly scores for the test set
test_weights = np.where(test_scores < 0, 1, 0)

# Print the weight values for the test set
print("Weight Values for the Test Set:")
print(test_weights)


# In[35]:



# Create a DataFrame with predictions
df_predictions = pd.DataFrame({'outliers': test_weights})

# Export the DataFrame to a CSV file
df_predictions.to_csv('predictions4.csv', index=False)


# In[ ]:




