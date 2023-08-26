#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Original Copy 6, the best one (?)

import pandas as pd
import numpy as np

# read in the data
data = pd.read_csv(r'C:\Users\Phoebe Tan\OneDrive - National University of Singapore\Desktop\Taiwan\Data Mining\HW2\train.csv', sep=',', header=0, skipinitialspace=True)
data.columns = [col.strip() for col in data.columns]


# In[2]:


data = data.replace('?', pd.NaT)


# In[3]:


data


# In[4]:


data.info()


# In[5]:


for col in ['workclass', 'occupation', 'native-country']:
    data[col].fillna(data[col].mode()[0], inplace=True)


# In[6]:


data.info()


# In[7]:


#Ploting the correlation between the output(income) and continuous features
data['income'] = data['income'].replace({'>50K': 1, '<=50K': 0})
import matplotlib.pyplot as plt
from matplotlib import rcParams
plt.matshow(data.corr())
plt.colorbar()
plt.xticks(np.arange(len(data.corr().columns)), data.corr().columns.values, rotation = 45) 
plt.yticks(np.arange(len(data.corr().columns)), data.corr().columns.values) 
for (i, j), corr in np.ndenumerate(data.corr()):
    plt.text(j, i, '{:0.1f}'.format(corr), ha='center', va='center', color='white', fontsize=14)
    
    
#Income has a correlation of 0 with fnlwgt, remove fnlwgt feature. #Drop fnlwgt


# In[8]:


data['age'].min()
data['age'].max()


# In[9]:


#Aggregating Age- Split age into groups of 10 

data['newage'] = pd.cut(data['age'], bins = [10,20,22,30,40,50,60,70,80,90])


# In[10]:


data.groupby(['newage','income']).size().unstack().plot(kind='bar', stacked=True)


# In[11]:


data['age'] = pd.cut(data['age'], bins = [0, 22, 50, 90], labels = ['Young', 'Adult', 'Old'])
data.drop('age', axis=1)


# In[12]:


data.groupby(['workclass','income']).size().unstack().plot(kind='bar', stacked=True)


# In[13]:


data.groupby(['education','income']).size().unstack().plot(kind='bar', stacked=True)


# In[14]:


#Group lower education levels together

data['education'].replace([ '10th', '11th','12th', '1st-4th', '5th-6th','7th-8th', '9th', 'Preschool' ],
                             ' Lowered', inplace = True)
data['education'].value_counts()


# In[15]:


data.groupby(['educational-num','income']).size().unstack().plot(kind='bar', stacked=True)


# In[16]:


# Checking for correlation between columns 'education' and 'education-num'
#Drop educational as it is similar to education-num

pd.crosstab(data['educational-num'],data['education'])


# In[17]:


data.groupby(['marital-status','income']).size().unstack().plot(kind='bar', stacked=True)


# In[18]:


# Create Married Column - Binary Yes(1) or No(0)
data["marital-status"] = data["marital-status"].replace(['Never-married','Divorced','Separated','Widowed'], 'Single')
data["marital-status"] = data["marital-status"].replace(['Married-civ-spouse','Married-spouse-absent','Married-AF-spouse'], 'Married')
data["marital-status"] = data["marital-status"].map({"Married":1, "Single":0})
data["marital-status"] = data["marital-status"].astype(int)


# In[19]:


data.groupby(['marital-status','income']).size().unstack().plot(kind='bar', stacked=True)


# In[20]:


data.groupby(['occupation','income']).size().unstack().plot(kind='bar', stacked=True)


# In[21]:


data.groupby(['relationship','income']).size().unstack().plot(kind='bar', stacked=True)


# In[22]:


#Group'Other-relative','Own-child','Unmarried' together

data['relationship'].replace([ 'Other-relative','Own-child','Unmarried'],
                             'otherelationship', inplace = True)
data['relationship'].value_counts()


# In[23]:


data.groupby(['race','income']).size().unstack().plot(kind='bar', stacked=True)


# In[24]:


#Group non-whites together

data['race'].replace([ 'Amer-Indian-Eskimo','Asian-Pac-Islander','Black','Other'],
                             'otherace', inplace = True)
data['race'].value_counts()


# In[25]:


data.groupby(['race','income']).size().unstack().plot(kind='bar', stacked=True)


# In[26]:


#Label Encoding for Race. 1-White, 0-Non-White
data['race'] = data['race'].replace({'White': 1, 'otherace': 0})


# In[27]:


data.groupby(['race','income']).size().unstack().plot(kind='bar', stacked=True)


# In[28]:


#Gender
data.groupby(['gender','income']).size().unstack().plot(kind='bar', stacked=True)


# In[29]:


#Label Encoding for Gender. 1-Female, Male
data['gender'] = data['gender'].replace({'Female': 1, 'Male': 0})


# In[30]:


len(data[['capital-gain','capital-loss']])


# In[31]:


#Capital Gain and Capital loss

data[['capital-gain','capital-loss']].value_counts()

#Majority of the values in capital-gain, and capital-loss are 0, Drop columns


# In[32]:


#Hours per week
data.groupby(['hours-per-week','income']).size().unstack().plot(kind='bar', stacked=True)


# In[33]:


data['hours-per-week'].value_counts()


# In[34]:


data['hours-per-week'].min()


# In[35]:


data['hours-per-week'].median()


# In[36]:


data['hours-per-week'].max()


# In[37]:


#Dividing hours of week in 3 major range and plotting it corresponding to the income

data['hours-per-week']= pd.cut(data['hours-per-week'], 
                                   bins = [0, 30, 50, 100], 
                                   labels = ['lowhours', 'normalhours', 'extrahours'])


# In[38]:


#Hours per week
data.groupby(['hours-per-week','income']).size().unstack().plot(kind='bar', stacked=True)


# In[39]:


#Country
data.groupby(['native-country','income']).size().unstack().plot(kind='bar', stacked=True)

#Lage Number of people from United States


# In[40]:


data['native-country'].value_counts(normalize=True)*100

#Majority of people from US. Drop column


# In[41]:


#Label Encoding for Native Country. 1-United States, 0-Others
#data['native-country'] = data['native-country'].apply(lambda x: 1 if x == 'United-States' else 0)


# In[42]:


#Country
#data.groupby(['native-country','income']).size().unstack().plot(kind='bar', stacked=True)


# In[43]:


data=data.drop(['capital-gain','capital-loss','native-country','fnlwgt','education'], axis = 1)


# In[44]:


data


# In[48]:


#Set Input: One Hot Encoding for categorical columns
workclass_df= pd.get_dummies(data['workclass'])
occupation_df=pd.get_dummies(data['occupation'])
rs_df=pd.get_dummies(data['relationship'])
hours_df=pd.get_dummies(data['hours-per-week'])
age_df=pd.get_dummies(data['age'])
other_df= data[['marital-status','race','gender','educational-num']]
#inputs= final_df.values


# In[49]:


final_df= workclass_df.join([occupation_df,rs_df,hours_df,age_df,other_df])


# In[50]:


final_df.info()


# In[51]:


# find unique values in column 'income'
unique_values = data['income'].unique()
print(unique_values)


# In[52]:


#Set Output
output_df= data['income']
outputs=data['income'].values


# In[53]:


#Join inputs and outputs, remove rwos with NA values
joined=final_df.join(output_df)


# In[54]:


# Create correlation matrix
corr_matrix = joined.corr().abs()



# In[55]:


# Select the absolute correlation values between 'income' and other features
income_corr = corr_matrix['income'].abs().sort_values(ascending=False)

# Exclude the 'income' column from the correlation values
income_corr = income_corr.drop('income')

# Select the top 10 features with the highest correlation values
top_features = income_corr.head(36).index.tolist()

# Print the top 10 features
print(top_features)


# In[56]:


income_corr


# In[225]:


#final_df2= final_df[top_features]
final_df2= final_df


# In[226]:


len(final_df2.columns)


# In[227]:



inputs= final_df2.values


# In[143]:


#Training and Evaluation (Cross Validation) WITH ALL 36 FEATURES, no regularization
# Define the number of splits, test size, and random state
n_splits = 5
test_size = 0.4
random_state = 0

# Create an array of indices for each data point
indices = np.arange(len(inputs))

# Store the accuracy scores for each fold
scores = []

# Define initial bias, weight, learning rate, iteration, and regularization parameter
b = 0.0
w = np.zeros(final_df2.shape[1])
lr = 0.01
iteration = 10000
reg_param = 0.1
batch_size = 350  # Set the batch size

# Normalize the features to have zero mean and unit variance
inputs = (inputs - np.mean(inputs, axis=0)) / np.std(inputs.astype(float), axis=0)

# Create a StratifiedShuffleSplit object
for i in range(n_splits):
    # Shuffle the indices to randomly select train and test indices
    np.random.shuffle(indices)

    # Calculate the number of test samples
    n_test = int(test_size * len(inputs))

    # Divide the indices into train and test sets
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    # Split the inputs and outputs into train and test sets using the selected indices
    X_train, X_test = inputs[train_indices], inputs[test_indices]
    y_train, y_test = outputs[train_indices], outputs[test_indices]

    # Perform mini-batch gradient descent
    for i in range(iteration):
        # Randomly select a batch of samples
        batch_indices = np.random.choice(X_train.shape[0], batch_size, replace=False)
        X_batch = X_train[batch_indices]
        y_batch = y_train[batch_indices]

        # Calculate the linear combination of features and weights for the batch
        z = np.dot(X_batch, w) + b

        # Apply the logistic function to convert the output to a probability value
        y_pred = 1 / (1 + np.exp(-z))

        # Calculate the gradient of the binary cross-entropy loss with respect to the weights and bias
        b_grad = np.mean(-(y_batch - y_pred))
        w_grad = np.mean(-(y_batch - y_pred)[:, None] * X_batch, axis=0)

        # Add regularization to the weight gradient
        #w_grad += 2*reg_param * w

        # Update the bias and weights using gradient descent
        b -= lr * b_grad
        w -= lr * w_grad

    # Calculate the predicted values using the final bias and weights
    z = np.dot(X_test, w) + b
    pred = 1 / (1 + np.exp(-z))

    # Map predicted probabilities to binary values based on threshold of 0.5
    pred = np.where(pred > 0.5, 1, 0)

    # Calculate accuracy score for the fold
    score = np.mean(pred == y_test)
    scores.append(score)

# Print the final bias and weight values
print('Final bias:', b)
print('Final weights:', w)

# Calculate the average accuracy score across all folds
print('Average accuracy score:', np.mean(scores))


# In[144]:



#iteration = 10000
#reg_param = 0.1
#batch_size = 350  # Set the batch size

#If i filtered based on correlation, accuracy score is: 0.8101861923347624 with regularization and 0.8283831339177171 without reg
#If i didn't filter based on correlation, acc is w reg: 0.8111331499136222 ,w/0 reg:  0.8310064623456395 (HIGHEST)


# In[145]:


#Testing out difference in accuracy with different number of features used (Based on correlation ranking) WITH REGULARIZATION
# Create an empty DataFrame with the specified columns
results = pd.DataFrame(columns=['index', 'accuracy', 'bias', 'weight'])

for h in range (37):
# Select the top h features with the highest correlation values
    top_features = income_corr.head(h).index.tolist()
    final_df2= final_df[top_features]
    inputs= final_df2.values
    #Training and Evaluation (Cross Validation)
    # Define the number of splits, test size, and random state
    n_splits = 5
    test_size = 0.4
    random_state = 0

    # Create an array of indices for each data point
    indices = np.arange(len(inputs))

    # Store the accuracy scores for each fold
    scores = []

    # Define initial bias, weight, learning rate, iteration, and regularization parameter
    b = 0.0
    w = np.zeros(final_df2.shape[1])
    lr = 0.01
    iteration = 10000
    reg_param = 0.1
    batch_size = 350  # Set the batch size

    # Normalize the features to have zero mean and unit variance
    inputs = (inputs - np.mean(inputs, axis=0)) / np.std(inputs.astype(float), axis=0)

    # Create a StratifiedShuffleSplit object
    for i in range(n_splits):
        # Shuffle the indices to randomly select train and test indices
        np.random.shuffle(indices)

        # Calculate the number of test samples
        n_test = int(test_size * len(inputs))

        # Divide the indices into train and test sets
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

        # Split the inputs and outputs into train and test sets using the selected indices
        X_train, X_test = inputs[train_indices], inputs[test_indices]
        y_train, y_test = outputs[train_indices], outputs[test_indices]

        # Perform mini-batch gradient descent
        for i in range(iteration):
            # Randomly select a batch of samples
            batch_indices = np.random.choice(X_train.shape[0], batch_size, replace=False)
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]

            # Calculate the linear combination of features and weights for the batch
            z = np.dot(X_batch, w) + b

            # Apply the logistic function to convert the output to a probability value
            y_pred = 1 / (1 + np.exp(-z))

            # Calculate the gradient of the binary cross-entropy loss with respect to the weights and bias
            b_grad = np.mean(-(y_batch - y_pred))
            w_grad = np.mean(-(y_batch - y_pred)[:, None] * X_batch, axis=0)

            # Add regularization to the weight gradient
            w_grad += 2*reg_param * w

            # Update the bias and weights using gradient descent
            b -= lr * b_grad
            w -= lr * w_grad

        # Calculate the predicted values using the final bias and weights
        z = np.dot(X_test, w) + b
        pred = 1 / (1 + np.exp(-z))

        # Map predicted probabilities to binary values based on threshold of 0.5
        pred = np.where(pred > 0.5, 1, 0)

        # Calculate accuracy score for the fold
        score = np.mean(pred == y_test)
        scores.append(score)
    results = results.append({'index': h, 'accuracy': np.mean(scores), 'bias': b, 'weight': w}, ignore_index=True)

    # Print the final bias and weight values
    #print('Final bias:', b)
    #print('Final weights:', w)

    # Calculate the average accuracy score across all folds
    print('Average accuracy score: ',h,' ', np.mean(scores))    
# Plot the accuracy over the iterations
plt.plot(results['index'], results['accuracy'])
plt.xlabel('Index')
plt.ylabel('Accuracy')
plt.show()       


# In[147]:


#Testing out difference in accuracy with different number of features used (Based on correlation ranking) WITHOUT REGULARIZATION
# Create an empty DataFrame with the specified columns
results = pd.DataFrame(columns=['index', 'accuracy', 'bias', 'weight'])

for h in range (37):
# Select the top h features with the highest correlation values
    top_features = income_corr.head(h).index.tolist()
    final_df2= final_df[top_features]
    inputs= final_df2.values
    #Training and Evaluation (Cross Validation)
    # Define the number of splits, test size, and random state
    n_splits = 5
    test_size = 0.4
    random_state = 0

    # Create an array of indices for each data point
    indices = np.arange(len(inputs))

    # Store the accuracy scores for each fold
    scores = []

    # Define initial bias, weight, learning rate, iteration, and regularization parameter
    b = 0.0
    w = np.zeros(final_df2.shape[1])
    lr = 0.01
    iteration = 10000
    reg_param = 0.1
    batch_size = 350  # Set the batch size

    # Normalize the features to have zero mean and unit variance
    inputs = (inputs - np.mean(inputs, axis=0)) / np.std(inputs.astype(float), axis=0)

    # Create a StratifiedShuffleSplit object
    for i in range(n_splits):
        # Shuffle the indices to randomly select train and test indices
        np.random.shuffle(indices)

        # Calculate the number of test samples
        n_test = int(test_size * len(inputs))

        # Divide the indices into train and test sets
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

        # Split the inputs and outputs into train and test sets using the selected indices
        X_train, X_test = inputs[train_indices], inputs[test_indices]
        y_train, y_test = outputs[train_indices], outputs[test_indices]

        # Perform mini-batch gradient descent
        for i in range(iteration):
            # Randomly select a batch of samples
            batch_indices = np.random.choice(X_train.shape[0], batch_size, replace=False)
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]

            # Calculate the linear combination of features and weights for the batch
            z = np.dot(X_batch, w) + b

            # Apply the logistic function to convert the output to a probability value
            y_pred = 1 / (1 + np.exp(-z))

            # Calculate the gradient of the binary cross-entropy loss with respect to the weights and bias
            b_grad = np.mean(-(y_batch - y_pred))
            w_grad = np.mean(-(y_batch - y_pred)[:, None] * X_batch, axis=0)

            # Add regularization to the weight gradient
            #w_grad += 2*reg_param * w

            # Update the bias and weights using gradient descent
            b -= lr * b_grad
            w -= lr * w_grad

        # Calculate the predicted values using the final bias and weights
        z = np.dot(X_test, w) + b
        pred = 1 / (1 + np.exp(-z))

        # Map predicted probabilities to binary values based on threshold of 0.5
        pred = np.where(pred > 0.5, 1, 0)

        # Calculate accuracy score for the fold
        score = np.mean(pred == y_test)
        scores.append(score)
    results = results.append({'index': h, 'accuracy': np.mean(scores), 'bias': b, 'weight': w}, ignore_index=True)

    # Print the final bias and weight values
    #print('Final bias:', b)
    #print('Final weights:', w)

    # Calculate the average accuracy score across all folds
    print('Average accuracy score: ',h,' ', np.mean(scores))    
# Plot the accuracy over the iterations
plt.plot(results['index'], results['accuracy'])
plt.xlabel('Index')
plt.ylabel('Accuracy')
plt.show()       


# In[59]:


# Select the top h features with the highest correlation values
top_features = income_corr.head(33).index.tolist()
final_df2= final_df[top_features]
inputs= final_df2.values

#Training and Evaluation (Cross Validation) WITH ALL 36 FEATURES, no regularization
# Define the number of splits, test size, and random state
n_splits = 5
test_size = 0.4
random_state = 0

# Create an array of indices for each data point
indices = np.arange(len(inputs))

# Store the accuracy scores for each fold
scores = []

# Define initial bias, weight, learning rate, iteration, and regularization parameter
b = 0.0
w = np.zeros(final_df2.shape[1])
lr = 0.01
iteration = 10000
reg_param = 0.1
batch_size = 350  # Set the batch size

# Normalize the features to have zero mean and unit variance
inputs = (inputs - np.mean(inputs, axis=0)) / np.std(inputs.astype(float), axis=0)

# Create a StratifiedShuffleSplit object
for i in range(n_splits):
    # Shuffle the indices to randomly select train and test indices
    np.random.shuffle(indices)

    # Calculate the number of test samples
    n_test = int(test_size * len(inputs))

    # Divide the indices into train and test sets
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    # Split the inputs and outputs into train and test sets using the selected indices
    X_train, X_test = inputs[train_indices], inputs[test_indices]
    y_train, y_test = outputs[train_indices], outputs[test_indices]

    # Perform mini-batch gradient descent
    for i in range(iteration):
        # Randomly select a batch of samples
        batch_indices = np.random.choice(X_train.shape[0], batch_size, replace=False)
        X_batch = X_train[batch_indices]
        y_batch = y_train[batch_indices]

        # Calculate the linear combination of features and weights for the batch
        z = np.dot(X_batch, w) + b

        # Apply the logistic function to convert the output to a probability value
        y_pred = 1 / (1 + np.exp(-z))

        # Calculate the gradient of the binary cross-entropy loss with respect to the weights and bias
        b_grad = np.mean(-(y_batch - y_pred))
        w_grad = np.mean(-(y_batch - y_pred)[:, None] * X_batch, axis=0)

        # Add regularization to the weight gradient
        #w_grad += 2*reg_param * w

        # Update the bias and weights using gradient descent
        b -= lr * b_grad
        w -= lr * w_grad

    # Calculate the predicted values using the final bias and weights
    z = np.dot(X_test, w) + b
    pred = 1 / (1 + np.exp(-z))

    # Map predicted probabilities to binary values based on threshold of 0.5
    pred = np.where(pred > 0.5, 1, 0)

    # Calculate accuracy score for the fold
    score = np.mean(pred == y_test)
    scores.append(score)

# Print the final bias and weight values
print('Final bias:', b)
print('Final weights:', w)

# Calculate the average accuracy score across all folds
print('Average accuracy score:', np.mean(scores))


# In[60]:


#Based on final model

# Create a dictionary with the feature names and weights
finalmodel = {'feature': final_df2.columns.tolist(), 'weights': w}

# Create a DataFrame from the dictionary
finalmodel = pd.DataFrame(finalmodel)

# Print the resulting DataFrame
print(finalmodel)



# In[61]:


#Based on final model, plot coefficient values and rank them

# Sort the DataFrame by the absolute value of weights in descending order
finalmodel = finalmodel.iloc[(finalmodel['weights'].abs()).argsort()]
fig, ax = plt.subplots(figsize=(10, 10))
# Create a horizontal bar chart of the weights
plt.barh(finalmodel['feature'], finalmodel['weights'].abs(), height=0.5)
plt.xlabel('Absolute Weight')
plt.ylabel('Feature')
plt.yticks(np.arange(len(finalmodel)), finalmodel['feature'],fontsize=11, va='center')
plt.show()


# In[229]:


#From above, CV score is highest when trained with 33 features; Here we are hypertuning of learning rate parameter using all features

import matplotlib.pyplot as plt
# Create an empty DataFrame with the specified columns
results = pd.DataFrame(columns=['rate', 'accuracy', 'bias', 'weight'])

# Select the absolute correlation values between 'income' and other features
income_corr = corr_matrix['income'].abs().sort_values(ascending=False)

# Exclude the 'income' column from the correlation values
income_corr = income_corr.drop('income')

top_features = income_corr.head(33).index.tolist()
final_df2= final_df[top_features]
inputs= final_df2.values

#Training and Evaluation (Cross Validation)
# Define the number of splits, test size, and random state
n_splits = 5
test_size = 0.4
random_state = 0

# Create an array of indices for each data point
indices = np.arange(len(inputs))

# Store the accuracy scores for each fold
scores = []

# Define initial bias, weight, learning rate, iteration, and regularization parameter
b = 0.0
w = np.zeros(final_df2.shape[1])
lr = 0.01
iteration = 10000
reg_param = 0.1
batch_size = 350  # Set the batch size

# Normalize the features to have zero mean and unit variance
inputs = (inputs - np.mean(inputs, axis=0)) / np.std(inputs.astype(float), axis=0)


for l in range(1,30,1):
    # Create a StratifiedShuffleSplit object
    lr= l/100
    for i in range(n_splits):
        # Shuffle the indices to randomly select train and test indices
        np.random.shuffle(indices)

        # Calculate the number of test samples
        n_test = int(test_size * len(inputs))

        # Divide the indices into train and test sets
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

        # Split the inputs and outputs into train and test sets using the selected indices
        X_train, X_test = inputs[train_indices], inputs[test_indices]
        y_train, y_test = outputs[train_indices], outputs[test_indices]

        # Perform mini-batch gradient descent
        for i in range(iteration):
            # Randomly select a batch of samples
            batch_indices = np.random.choice(X_train.shape[0], batch_size, replace=False)
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]

            # Calculate the linear combination of features and weights for the batch
            z = np.dot(X_batch, w) + b

            # Apply the logistic function to convert the output to a probability value
            y_pred = 1 / (1 + np.exp(-z))

            # Calculate the gradient of the binary cross-entropy loss with respect to the weights and bias
            b_grad = np.mean(-(y_batch - y_pred))
            w_grad = np.mean(-(y_batch - y_pred)[:, None] * X_batch, axis=0)

            # Add regularization to the weight gradient
            #w_grad += 2*reg_param * w

            # Update the bias and weights using gradient descent
            b -= lr * b_grad
            w -= lr * w_grad

        # Calculate the predicted values using the final bias and weights
        z = np.dot(X_test, w) + b
        pred = 1 / (1 + np.exp(-z))

        # Map predicted probabilities to binary values based on threshold of 0.5
        pred = np.where(pred > 0.5, 1, 0)

        # Calculate accuracy score for the fold
        score = np.mean(pred == y_test)
        scores.append(score)

    # Print the final bias and weight values
    #print('Final bias:', b)
    #print('Final weights:', w)
    results = results.append({'rate': lr, 'accuracy': np.mean(scores), 'bias': b, 'weight': w}, ignore_index=True)
    # Calculate the average accuracy score across all folds
    print('Average accuracy score:', lr, np.mean(scores))
# Plot the accuracy over the iterations
plt.plot(results['rate'], results['accuracy'])
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.show()   


# In[57]:


#Train model by removing one feature each time, without regularization

import matplotlib.pyplot as plt
# Create an empty DataFrame with the specified columns
results = pd.DataFrame(columns=['index', 'accuracy', 'bias', 'weight'])


for h in range(36): #h is the removed feature
    # Remove one feature each time
    top_features=income_corr.index.tolist()[:h]+income_corr.index.tolist()[h+1:]
    final_df2= final_df[top_features]
    inputs= final_df2.values
    #Training and Evaluation (Cross Validation)
    # Define the number of splits, test size, and random state
    n_splits = 5
    test_size = 0.4
    random_state = 0

    # Create an array of indices for each data point
    indices = np.arange(len(inputs))

    # Store the accuracy scores for each fold
    scores = []

    # Define initial bias, weight, learning rate, iteration, and regularization parameter
    b = 0.0
    w = np.zeros(final_df2.shape[1])
    lr = 0.01
    iteration = 10000
    reg_param = 0.1
    batch_size = 350  # Set the batch size

    # Normalize the features to have zero mean and unit variance
    inputs = (inputs - np.mean(inputs, axis=0)) / np.std(inputs.astype(float), axis=0)

    # Create a StratifiedShuffleSplit object
    for i in range(n_splits):
        # Shuffle the indices to randomly select train and test indices
        np.random.shuffle(indices)

        # Calculate the number of test samples
        n_test = int(test_size * len(inputs))

        # Divide the indices into train and test sets
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

        # Split the inputs and outputs into train and test sets using the selected indices
        X_train, X_test = inputs[train_indices], inputs[test_indices]
        y_train, y_test = outputs[train_indices], outputs[test_indices]

        # Perform mini-batch gradient descent
        for i in range(iteration):
            # Randomly select a batch of samples
            batch_indices = np.random.choice(X_train.shape[0], batch_size, replace=False)
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]

            # Calculate the linear combination of features and weights for the batch
            z = np.dot(X_batch, w) + b

            # Apply the logistic function to convert the output to a probability value
            y_pred = 1 / (1 + np.exp(-z))

            # Calculate the gradient of the binary cross-entropy loss with respect to the weights and bias
            b_grad = np.mean(-(y_batch - y_pred))
            w_grad = np.mean(-(y_batch - y_pred)[:, None] * X_batch, axis=0)

            # Add regularization to the weight gradient
            #w_grad += 2*reg_param * w

            # Update the bias and weights using gradient descent
            b -= lr * b_grad
            w -= lr * w_grad

        # Calculate the predicted values using the final bias and weights
        z = np.dot(X_test, w) + b
        pred = 1 / (1 + np.exp(-z))

        # Map predicted probabilities to binary values based on threshold of 0.5
        pred = np.where(pred > 0.5, 1, 0)

        # Calculate accuracy score for the fold
        score = np.mean(pred == y_test)
        scores.append(score)
        
    results = results.append({'index': h, 'accuracy': np.mean(scores), 'bias': b, 'weight': w}, ignore_index=True)
    # Print the final bias and weight values
    #print('Final bias:', b)
    #print('Final weights:', w)

    # Calculate the average accuracy score across all folds
    print('Average accuracy score: ',h,' ', np.mean(scores))    
    
    
    
# Plot the accuracy over the iterations
plt.plot(results['index'], results['accuracy'])
plt.xlabel('Index')
plt.ylabel('Accuracy')
plt.show()       

    


# In[ ]:


#Train model by removing one feature each time, with regularization



import matplotlib.pyplot as plt
# Create an empty DataFrame with the specified columns
results = pd.DataFrame(columns=['index', 'accuracy', 'bias', 'weight'])


for h in range(36): #h is the removed feature
    # Remove one feature each time
    top_features=income_corr.index.tolist()[:h]+income_corr.index.tolist()[h+1:]
    final_df2= final_df[top_features]
    inputs= final_df2.values
    #Training and Evaluation (Cross Validation)
    # Define the number of splits, test size, and random state
    n_splits = 5
    test_size = 0.4
    random_state = 0

    # Create an array of indices for each data point
    indices = np.arange(len(inputs))

    # Store the accuracy scores for each fold
    scores = []

    # Define initial bias, weight, learning rate, iteration, and regularization parameter
    b = 0.0
    w = np.zeros(final_df2.shape[1])
    lr = 0.01
    iteration = 10000
    reg_param = 0.1
    batch_size = 350  # Set the batch size

    # Normalize the features to have zero mean and unit variance
    inputs = (inputs - np.mean(inputs, axis=0)) / np.std(inputs.astype(float), axis=0)

    # Create a StratifiedShuffleSplit object
    for i in range(n_splits):
        # Shuffle the indices to randomly select train and test indices
        np.random.shuffle(indices)

        # Calculate the number of test samples
        n_test = int(test_size * len(inputs))

        # Divide the indices into train and test sets
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

        # Split the inputs and outputs into train and test sets using the selected indices
        X_train, X_test = inputs[train_indices], inputs[test_indices]
        y_train, y_test = outputs[train_indices], outputs[test_indices]

        # Perform mini-batch gradient descent
        for i in range(iteration):
            # Randomly select a batch of samples
            batch_indices = np.random.choice(X_train.shape[0], batch_size, replace=False)
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]

            # Calculate the linear combination of features and weights for the batch
            z = np.dot(X_batch, w) + b

            # Apply the logistic function to convert the output to a probability value
            y_pred = 1 / (1 + np.exp(-z))

            # Calculate the gradient of the binary cross-entropy loss with respect to the weights and bias
            b_grad = np.mean(-(y_batch - y_pred))
            w_grad = np.mean(-(y_batch - y_pred)[:, None] * X_batch, axis=0)

            # Add regularization to the weight gradient
            w_grad += 2*reg_param * w

            # Update the bias and weights using gradient descent
            b -= lr * b_grad
            w -= lr * w_grad

        # Calculate the predicted values using the final bias and weights
        z = np.dot(X_test, w) + b
        pred = 1 / (1 + np.exp(-z))

        # Map predicted probabilities to binary values based on threshold of 0.5
        pred = np.where(pred > 0.5, 1, 0)

        # Calculate accuracy score for the fold
        score = np.mean(pred == y_test)
        scores.append(score)
    
    results = results.append({'index': h, 'accuracy': np.mean(scores), 'bias': b, 'weight': w}, ignore_index=True)
    # Print the final bias and weight values
    #print('Final bias:', b)
    #print('Final weights:', w)

    # Calculate the average accuracy score across all folds
    print('Average accuracy score: ',h,' ', np.mean(scores))    
    
    
# Plot the accuracy over the iterations
plt.plot(results['index'], results['accuracy'])
plt.xlabel('Index')
plt.ylabel('Accuracy')
plt.show()       
    
    

    


# In[150]:


#Preprocess Testing Data

# read in the data
test = pd.read_csv(r'C:\Users\Phoebe Tan\OneDrive - National University of Singapore\Desktop\Taiwan\Data Mining\HW2\test_X.csv', sep=',', header=0, skipinitialspace=True)
test.columns = [col.strip() for col in test.columns]
    


# In[151]:


test = test.replace('?', pd.NaT)


# In[152]:


#Remove rows with NA values
# Find the row indices with at least one NaN value
nan_rows = test.isna().any(axis=1)

# Print the number of rows with NA values
print('Number of rows with NA values:',sum(nan_rows==True))

#There are no rows with NA values


# In[153]:


test.info()


# In[154]:


for col in ['workclass', 'occupation', 'native-country']:
    test[col].fillna(test[col].mode()[0], inplace=True)


# In[155]:


test.info()


# In[156]:


data=test


# In[157]:


data['age'] = pd.cut(data['age'], bins = [0, 22, 50, 90], labels = ['Young', 'Adult', 'Old'])
data.drop('age', axis=1)


#Group lower education levels together

data['education'].replace([ '10th', '11th','12th', '1st-4th', '5th-6th','7th-8th', '9th', 'Preschool' ],
                             ' Lowered', inplace = True)
data['education'].value_counts()


# Create Married Column - Binary Yes(1) or No(0)
data["marital-status"] = data["marital-status"].replace(['Never-married','Divorced','Separated','Widowed'], 'Single')
data["marital-status"] = data["marital-status"].replace(['Married-civ-spouse','Married-spouse-absent','Married-AF-spouse'], 'Married')
data["marital-status"] = data["marital-status"].map({"Married":1, "Single":0})
data["marital-status"] = data["marital-status"].astype(int)


#Group married spouse absemt, married af spouse, separated, widowed together

data['relationship'].replace([ 'Other-relative','Own-child','Unmarried'],
                             'otherelationship', inplace = True)
data['relationship'].value_counts()

#Group non-whites together

data['race'].replace([ 'Amer-Indian-Eskimo','Asian-Pac-Islander','Black','Other'],
                             'otherace', inplace = True)
data['race'].value_counts()

#Label Encoding for Race. 1-White, 0-Non-White
data['race'] = data['race'].replace({'White': 1, 'otherace': 0})

#Label Encoding for Gender. 1-Female, Male
data['gender'] = data['gender'].replace({'Female': 1, 'Male': 0})


#Hours per week
data['hours-per-week']= pd.cut(data['hours-per-week'], 
                                   bins = [0, 30, 50, 100], 
                                   labels = ['lowhours', 'normalhours', 'extrahours'])

data=data.drop(['capital-gain','capital-loss','native-country','fnlwgt','education'], axis = 1)


#Set Input: One Hot Encoding for categorical columns
workclass_df= pd.get_dummies(data['workclass'])
occupation_df=pd.get_dummies(data['occupation'])
rs_df=pd.get_dummies(data['relationship'])
hours_df=pd.get_dummies(data['hours-per-week'])
age_df=pd.get_dummies(data['age'])
other_df= data[['marital-status','race','gender','educational-num']]
#inputs= final_df.values

final_df= workclass_df.join([occupation_df,rs_df,hours_df,age_df,other_df])


# In[162]:



final_df2= final_df


# In[163]:


final_testdf=final_df2[top_features]


# In[164]:


#Remove rows with NA values
# Find the row indices with at least one NaN value
nan_rows = final_testdf.isna().any(axis=1)

# Print the number of rows with NA values
print('Number of rows with NA values:',sum(nan_rows==True))

#There are no rows with NA values


# In[165]:


X = np.array(final_testdf)

# Normalize the features to have zero mean and unit variance
X = (X - np.mean(X, axis=0)) / np.std(X.astype(float), axis=0)


# In[166]:


X


# In[167]:


#Initialize y predicted values
predicted= np.zeros(X.shape[0])


for j in range(X.shape[0]):
    z = np.dot(w, X[j]) + b
    pred = 1 / (1 + np.exp(-z))
    # Map predicted probabilities to binary values based on threshold of 0.5
    pred = np.where(pred > 0.5, 1, 0)
    predicted[j]=pred


# Print the final bias and weight values
print('Final bias:', b)
print('Final weights:', w)


# In[168]:


predicted


# In[169]:


import csv
# Transpose the array
predicted = np.transpose(predicted)



# Open the CSV file in write mode
with open('outputfinalls.csv', mode='w', newline='') as file:

    # Create a CSV writer object
    writer = csv.writer(file)

    # Write the array as a row
    writer.writerow(predicted)


# In[ ]:





# In[ ]:





# In[ ]:




