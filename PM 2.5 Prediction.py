#!/usr/bin/env python
# coding: utf-8

# In[443]:


import pandas as pd
import numpy as np

# read in the data
data = pd.read_csv(r"C:\Users\Phoebe Tan\OneDrive - National University of Singapore\Desktop\Taiwan\Data Mining\train.csv", sep=',', header=0, skipinitialspace=True)
data.columns = [col.strip() for col in data.columns]
data['ItemName']=data['ItemName'].str.strip()

# Load the data from CSV
df=data

# Convert the 'Date' column to a datetime format
df['Date'] = pd.to_datetime(df['Date'])


# In[444]:


# replace the invalid values with NaN
df = df.replace(['AMB_TEMP'], 'TEMP', regex=True)
df = df.replace(['RAINFALL'], 'RF', regex=True)
df = df.replace(['NOx'], 'NO3', regex=True)
df = df.replace(['#\s*', '\*\s*', 'x\s*', 'A\s*'], pd.NA, regex=True)
df = df.replace(['TEMP'], 'AMB_TEMP', regex=True)
df = df.replace(['NO3'], 'NOX', regex=True)
df = df.replace(['RF'], 'RAINFALL', regex=True)


# In[445]:


df[['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23']]= df[['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23']].apply(pd.to_numeric)


# In[446]:


df


# In[447]:


# Extract the month from the 'date' column
df['Month'] = df['Date'].dt.month
print(df)


# In[448]:


# Find the row indices with at least one NaN value
nan_rows = df.isna().any(axis=1)

# Print the number of rows with NA values
print('Number of rows with NA values:',sum(nan_rows==True))


# In[449]:


#HANDLING NA VALUES

# Iterate over each unique item in the original dataframe
for item in df['ItemName'].unique():
    for month in df['Month'].unique():
        # Get the rows for the current item and month
        item_month_rows = df[(df['ItemName'] == item) & (df['Month'] == month)]
        prevrow=-1

        # Fill in missing values in each row
        for i, row in item_month_rows.iterrows():
            # Get the boolean mask of missing values in the row
            missing_mask = row.isna()

            # Fill in first column missing value with previous row last column value
            if missing_mask['0']:
                if prevrow==-1:
                    row['0'] = row['23']
                    df.at[i, '0'] = row['0']

                else:
                    prev_row = df.iloc[prevrow]
                    row['0'] = prev_row['23']
                    df.at[i, '0'] = row['0']


            # Fill in missing values with value from the previous column
            for j in range(1,24):
                if missing_mask[str(j)]:
                    row[str(j)] = row[str(j-1)]
                    df.at[i, str(j)] = row[str(j)]
            prevrow=i


# In[450]:


# Find the row indices with at least one NaN value
nan_rows = df.isna().any(axis=1)

# Print the number of rows with NA values
print('Number of rows with NA values:',sum(nan_rows==True))


# In[451]:


#df.to_csv("fdafsdddddddddddf.csv", index=False)
df


# In[452]:


#Separate by month

month1 = df[df['Month'] == 1]
month2 = df[df['Month'] == 2]
month3 = df[df['Month'] == 3]
month4 = df[df['Month'] == 4]
month5 = df[df['Month'] == 5]
month6 = df[df['Month'] == 6]
month7 = df[df['Month'] == 7]
month8 = df[df['Month'] == 8]
month9 = df[df['Month'] == 9]
month10 = df[df['Month'] ==10]
month11 = df[df['Month'] == 11]
month12 = df[df['Month'] == 12]


# In[453]:


originaldf=df
originaldf


# month1

# In[454]:


# iterate over each unique item in the original dataframe
df=month1.drop(["Month","Location","Date"], axis=1)
data_dict={}
for item in df['ItemName'].unique():
    # get the rows for the current item
    item_rows = df[df['ItemName'] == item]
    tempvalues=[]
    data_dict[item]=tempvalues
    for i, row in item_rows.iterrows():
        for col in item_rows.columns[1:]:
            tempvalues.append(row[col])
    
# Define the size of the sliding window
window_size = 10

# Initialize an empty list to hold the results
result_list = []

# Iterate over each key-value pair in the dictionary
for item_name, values in data_dict.items():
    # Iterate over each starting index for the window
    for start_idx in range(len(values) - window_size + 1):
        # Get the values for the current window
        window_values = values[start_idx : start_idx + window_size]
        
        # Append the results to the result list
        result_list.append([item_name] + window_values)

# Convert the results into a pandas DataFrame
column_names = ['ItemName'] + [f'Hour {i+1}' for i in range(window_size)]
mth1 = pd.DataFrame(result_list, columns=column_names)

# Print the resulting DataFrame
print(mth1)
#result_df.to_csv("month1.00.csv", index=False)


# In[455]:


# iterate over each unique item in the original dataframe
df=month2.drop(["Month","Location","Date"], axis=1)
data_dict={}
for item in df['ItemName'].unique():
    # get the rows for the current item
    item_rows = df[df['ItemName'] == item]
    tempvalues=[]
    data_dict[item]=tempvalues
    for i, row in item_rows.iterrows():
        for col in item_rows.columns[1:]:
            tempvalues.append(row[col])
    
# Define the size of the sliding window
window_size = 10

# Initialize an empty list to hold the results
result_list = []

# Iterate over each key-value pair in the dictionary
for item_name, values in data_dict.items():
    # Iterate over each starting index for the window
    for start_idx in range(len(values) - window_size + 1):
        # Get the values for the current window
        window_values = values[start_idx : start_idx + window_size]
        
        # Append the results to the result list
        result_list.append([item_name] + window_values)

# Convert the results into a pandas DataFrame
column_names = ['ItemName'] + [f'Hour {i+1}' for i in range(window_size)]
mth2 = pd.DataFrame(result_list, columns=column_names)

# Print the resulting DataFrame
print(mth2)
#result_df.to_csv("month1.00.csv", index=False)


# In[456]:


# iterate over each unique item in the original dataframe
df=month3.drop(["Month","Location","Date"], axis=1)
data_dict={}
for item in df['ItemName'].unique():
    # get the rows for the current item
    item_rows = df[df['ItemName'] == item]
    tempvalues=[]
    data_dict[item]=tempvalues
    for i, row in item_rows.iterrows():
        for col in item_rows.columns[1:]:
            tempvalues.append(row[col])
    
# Define the size of the sliding window
window_size = 10

# Initialize an empty list to hold the results
result_list = []

# Iterate over each key-value pair in the dictionary
for item_name, values in data_dict.items():
    # Iterate over each starting index for the window
    for start_idx in range(len(values) - window_size + 1):
        # Get the values for the current window
        window_values = values[start_idx : start_idx + window_size]
        
        # Append the results to the result list
        result_list.append([item_name] + window_values)

# Convert the results into a pandas DataFrame
column_names = ['ItemName'] + [f'Hour {i+1}' for i in range(window_size)]
mth3 = pd.DataFrame(result_list, columns=column_names)

# Print the resulting DataFrame
print(mth3)
#result_df.to_csv("month1.00.csv", index=False)


# In[457]:


# iterate over each unique item in the original dataframe
df=month4.drop(["Month","Location","Date"], axis=1)
data_dict={}
for item in df['ItemName'].unique():
    # get the rows for the current item
    item_rows = df[df['ItemName'] == item]
    tempvalues=[]
    data_dict[item]=tempvalues
    for i, row in item_rows.iterrows():
        for col in item_rows.columns[1:]:
            tempvalues.append(row[col])
    
# Define the size of the sliding window
window_size = 10

# Initialize an empty list to hold the results
result_list = []

# Iterate over each key-value pair in the dictionary
for item_name, values in data_dict.items():
    # Iterate over each starting index for the window
    for start_idx in range(len(values) - window_size + 1):
        # Get the values for the current window
        window_values = values[start_idx : start_idx + window_size]
        
        # Append the results to the result list
        result_list.append([item_name] + window_values)

# Convert the results into a pandas DataFrame
column_names = ['ItemName'] + [f'Hour {i+1}' for i in range(window_size)]
mth4 = pd.DataFrame(result_list, columns=column_names)

# Print the resulting DataFrame
print(mth4)
#result_df.to_csv("month1.00.csv", index=False)


# In[458]:


# iterate over each unique item in the original dataframe
df=month5.drop(["Month","Location","Date"], axis=1)
data_dict={}
for item in df['ItemName'].unique():
    # get the rows for the current item
    item_rows = df[df['ItemName'] == item]
    tempvalues=[]
    data_dict[item]=tempvalues
    for i, row in item_rows.iterrows():
        for col in item_rows.columns[1:]:
            tempvalues.append(row[col])
    
# Define the size of the sliding window
window_size = 10

# Initialize an empty list to hold the results
result_list = []

# Iterate over each key-value pair in the dictionary
for item_name, values in data_dict.items():
    # Iterate over each starting index for the window
    for start_idx in range(len(values) - window_size + 1):
        # Get the values for the current window
        window_values = values[start_idx : start_idx + window_size]
        
        # Append the results to the result list
        result_list.append([item_name] + window_values)

# Convert the results into a pandas DataFrame
column_names = ['ItemName'] + [f'Hour {i+1}' for i in range(window_size)]
mth5 = pd.DataFrame(result_list, columns=column_names)

# Print the resulting DataFrame
print(mth5)
#result_df.to_csv("month1.00.csv", index=False)


# In[459]:


# iterate over each unique item in the original dataframe
df=month6.drop(["Month","Location","Date"], axis=1)
data_dict={}
for item in df['ItemName'].unique():
    # get the rows for the current item
    item_rows = df[df['ItemName'] == item]
    tempvalues=[]
    data_dict[item]=tempvalues
    for i, row in item_rows.iterrows():
        for col in item_rows.columns[1:]:
            tempvalues.append(row[col])
    
# Define the size of the sliding window
window_size = 10

# Initialize an empty list to hold the results
result_list = []

# Iterate over each key-value pair in the dictionary
for item_name, values in data_dict.items():
    # Iterate over each starting index for the window
    for start_idx in range(len(values) - window_size + 1):
        # Get the values for the current window
        window_values = values[start_idx : start_idx + window_size]
        
        # Append the results to the result list
        result_list.append([item_name] + window_values)

# Convert the results into a pandas DataFrame
column_names = ['ItemName'] + [f'Hour {i+1}' for i in range(window_size)]
mth6 = pd.DataFrame(result_list, columns=column_names)

# Print the resulting DataFrame
print(mth6)
#result_df.to_csv("month1.00.csv", index=False)


# In[460]:


# iterate over each unique item in the original dataframe
df=month7.drop(["Month","Location","Date"], axis=1)
data_dict={}
for item in df['ItemName'].unique():
    # get the rows for the current item
    item_rows = df[df['ItemName'] == item]
    tempvalues=[]
    data_dict[item]=tempvalues
    for i, row in item_rows.iterrows():
        for col in item_rows.columns[1:]:
            tempvalues.append(row[col])
    
# Define the size of the sliding window
window_size = 10

# Initialize an empty list to hold the results
result_list = []

# Iterate over each key-value pair in the dictionary
for item_name, values in data_dict.items():
    # Iterate over each starting index for the window
    for start_idx in range(len(values) - window_size + 1):
        # Get the values for the current window
        window_values = values[start_idx : start_idx + window_size]
        
        # Append the results to the result list
        result_list.append([item_name] + window_values)

# Convert the results into a pandas DataFrame
column_names = ['ItemName'] + [f'Hour {i+1}' for i in range(window_size)]
mth7 = pd.DataFrame(result_list, columns=column_names)

# Print the resulting DataFrame
print(mth7)
#result_df.to_csv("month1.00.csv", index=False)


# In[461]:


# iterate over each unique item in the original dataframe
df=month8.drop(["Month","Location","Date"], axis=1)
data_dict={}
for item in df['ItemName'].unique():
    # get the rows for the current item
    item_rows = df[df['ItemName'] == item]
    tempvalues=[]
    data_dict[item]=tempvalues
    for i, row in item_rows.iterrows():
        for col in item_rows.columns[1:]:
            tempvalues.append(row[col])
    
# Define the size of the sliding window
window_size = 10

# Initialize an empty list to hold the results
result_list = []

# Iterate over each key-value pair in the dictionary
for item_name, values in data_dict.items():
    # Iterate over each starting index for the window
    for start_idx in range(len(values) - window_size + 1):
        # Get the values for the current window
        window_values = values[start_idx : start_idx + window_size]
        
        # Append the results to the result list
        result_list.append([item_name] + window_values)

# Convert the results into a pandas DataFrame
column_names = ['ItemName'] + [f'Hour {i+1}' for i in range(window_size)]
mth8 = pd.DataFrame(result_list, columns=column_names)

# Print the resulting DataFrame
print(mth8)
#result_df.to_csv("month1.00.csv", index=False)


# In[462]:


# iterate over each unique item in the original dataframe
df=month9.drop(["Month","Location","Date"], axis=1)
data_dict={}
for item in df['ItemName'].unique():
    # get the rows for the current item
    item_rows = df[df['ItemName'] == item]
    tempvalues=[]
    data_dict[item]=tempvalues
    for i, row in item_rows.iterrows():
        for col in item_rows.columns[1:]:
            tempvalues.append(row[col])
    
# Define the size of the sliding window
window_size = 10

# Initialize an empty list to hold the results
result_list = []

# Iterate over each key-value pair in the dictionary
for item_name, values in data_dict.items():
    # Iterate over each starting index for the window
    for start_idx in range(len(values) - window_size + 1):
        # Get the values for the current window
        window_values = values[start_idx : start_idx + window_size]
        
        # Append the results to the result list
        result_list.append([item_name] + window_values)

# Convert the results into a pandas DataFrame
column_names = ['ItemName'] + [f'Hour {i+1}' for i in range(window_size)]
mth9 = pd.DataFrame(result_list, columns=column_names)

# Print the resulting DataFrame
print(mth9)
#result_df.to_csv("month1.00.csv", index=False)


# In[463]:


# iterate over each unique item in the original dataframe
df=month10.drop(["Month","Location","Date"], axis=1)
data_dict={}
for item in df['ItemName'].unique():
    # get the rows for the current item
    item_rows = df[df['ItemName'] == item]
    tempvalues=[]
    data_dict[item]=tempvalues
    for i, row in item_rows.iterrows():
        for col in item_rows.columns[1:]:
            tempvalues.append(row[col])
    
# Define the size of the sliding window
window_size = 10

# Initialize an empty list to hold the results
result_list = []

# Iterate over each key-value pair in the dictionary
for item_name, values in data_dict.items():
    # Iterate over each starting index for the window
    for start_idx in range(len(values) - window_size + 1):
        # Get the values for the current window
        window_values = values[start_idx : start_idx + window_size]
        
        # Append the results to the result list
        result_list.append([item_name] + window_values)

# Convert the results into a pandas DataFrame
column_names = ['ItemName'] + [f'Hour {i+1}' for i in range(window_size)]
mth10 = pd.DataFrame(result_list, columns=column_names)

# Print the resulting DataFrame
print(mth10)
#result_df.to_csv("month1.00.csv", index=False)


# In[464]:


# iterate over each unique item in the original dataframe
df=month11.drop(["Month","Location","Date"], axis=1)
data_dict={}
for item in df['ItemName'].unique():
    # get the rows for the current item
    item_rows = df[df['ItemName'] == item]
    tempvalues=[]
    data_dict[item]=tempvalues
    for i, row in item_rows.iterrows():
        for col in item_rows.columns[1:]:
            tempvalues.append(row[col])
    
# Define the size of the sliding window
window_size = 10

# Initialize an empty list to hold the results
result_list = []

# Iterate over each key-value pair in the dictionary
for item_name, values in data_dict.items():
    # Iterate over each starting index for the window
    for start_idx in range(len(values) - window_size + 1):
        # Get the values for the current window
        window_values = values[start_idx : start_idx + window_size]
        
        # Append the results to the result list
        result_list.append([item_name] + window_values)

# Convert the results into a pandas DataFrame
column_names = ['ItemName'] + [f'Hour {i+1}' for i in range(window_size)]
mth11 = pd.DataFrame(result_list, columns=column_names)

# Print the resulting DataFrame
print(mth11)
#result_df.to_csv("month1.00.csv", index=False)


# In[465]:


# iterate over each unique item in the original dataframe
df=month12.drop(["Month","Location","Date"], axis=1)
data_dict={}
for item in df['ItemName'].unique():
    # get the rows for the current item
    item_rows = df[df['ItemName'] == item]
    tempvalues=[]
    data_dict[item]=tempvalues
    for i, row in item_rows.iterrows():
        for col in item_rows.columns[1:]:
            tempvalues.append(row[col])
    
# Define the size of the sliding window
window_size = 10

# Initialize an empty list to hold the results
result_list = []

# Iterate over each key-value pair in the dictionary
for item_name, values in data_dict.items():
    # Iterate over each starting index for the window
    for start_idx in range(len(values) - window_size + 1):
        # Get the values for the current window
        window_values = values[start_idx : start_idx + window_size]
        
        # Append the results to the result list
        result_list.append([item_name] + window_values)

# Convert the results into a pandas DataFrame
column_names = ['ItemName'] + [f'Hour {i+1}' for i in range(window_size)]
mth12 = pd.DataFrame(result_list, columns=column_names)

# Print the resulting DataFrame
print(mth12)
#result_df.to_csv("month1.00.csv", index=False)


# In[466]:


# Concatenate the DataFrames vertically
df= pd.concat([mth1,mth2,mth3,mth4,mth5,mth6,mth7,mth8,mth9,mth10,mth11,mth12])

# Print the resulting DataFrame
print(df)


# In[529]:


# calculate the mean and median of the Hour 1 to 9 columns
df["Mean"] = df.iloc[:, 1:9].mean(axis=1)
df["Median"] = df.iloc[:, 1:9].median(axis=1)

# calculate the means for each subset of columns
df["Mean1"] = df.iloc[:, 1:3].mean(axis=1)
df["Mean2"] = df.iloc[:, 4:6].mean(axis=1)
df["Mean3"] = df.iloc[:, 7:9].mean(axis=1)

df


# In[530]:


# Label each group with an ID from 0 to the length of the group
df['id'] = df.groupby('ItemName').cumcount()

# Print the resulting DataFrame
print(df)


# In[531]:


# Feature selection: Aggregation (Method 1- Take 2) Take mean of the 3hours data as a feature for each variable to predict 10 hours)

# Create a list of column names
columns = ['AMB_TEMPa', 'CH4a', 'COa', 'NMHCa', 'NOa', 'NO2a', 'NOXa', 'O3a', 'PM10a', 'PM2.5a','RAINFALLa', 'RHa', 'SO2a', 'THCa', 'WD_HRa', 'WIND_DIRECa', 'WIND_SPEEDa','WS_HRa','AMB_TEMPb', 'CH4b', 'COb', 'NMHCb', 'NOb', 'NO2b', 'NOXb', 'O3b', 'PM10b', 'PM2.5b','RAINFALLb', 'RHb', 'SO2b', 'THCb', 'WD_HRb', 'WIND_DIRECb', 'WIND_SPEEDb','WS_HRb','AMB_TEMPc', 'CH4c', 'COc', 'NMHCc', 'NOc', 'NO2c', 'NOXc', 'O3c', 'PM10c', 'PM2.5c','RAINFALLc', 'RHc', 'SO2c', 'THCc', 'WD_HRc', 'WIND_DIRECc', 'WIND_SPEEDc','WS_HRc','Output']
# Create an empty DataFrame with the specified columns
method1 = pd.DataFrame(columns=columns)
method1['Output']=pd.to_numeric(method1["Output"])

# insert Id column at beginning
method1.insert(0, 'id', range(0, 5652))


# iterate through each id
for i in range(5652):
    id_df = df.loc[df['id'] == i] # select rows with the iterated id value
    for index, row in id_df.iterrows():
        item_name = row['ItemName']
        mean1 =row['Mean1']
        mean2 =row['Mean2']
        mean3=row['Mean3']
        method1.loc[method1['id'] == i,item_name +('a')] = mean1 
        method1.loc[method1['id'] == i,item_name+('b')] = mean2
        method1.loc[method1['id'] == i,item_name+('c')] = mean3     
        # add PM2.5 value in Hour 10 of the iterated index to the new dataset column 'Output'
        if item_name=="PM2.5":
            pm25_value=row['Hour 10']
            method1.loc[method1['id'] == i, 'Output'] = pm25_value
        else:
            continue
   
        
# print the new dataset
print(method1)


# In[573]:


#Reduced data 
reduced_data = method1.drop(['id','Output'],axis=1)
reduced_data = np.array(reduced_data)


# In[574]:


reduced_data


# In[575]:


#Performing Linear Regression
y  = method1['Output']
y = np.array(y)
X= reduced_data

pred = np.zeros(X.shape[0])
# Normalize the features to have zero mean and unit variance
X = (X - np.mean(X, axis=0)) / np.std(X.astype(float), axis=0)
# Define initial bias, weight, learning rate, iteration, and regularization parameter
b = 0.0
w = np.zeros(X.shape[1])
lr = 0.03
iteration = 8000000
reg_param = 0.1

# Initialize Adagrad learning rate parameters
lr_b = 0.0
lr_w = np.zeros(X.shape[1])

# Perform stochastic gradient descent with Adagrad
for i in range(iteration):
    # Randomly select a single sample
    j = np.random.randint(X.shape[0])
    
    y_pred = np.dot(w, X[j]) + b
    b_grad = -2.0*(y[j] - y_pred) * 1
    w_grad = -2.0*(y[j] - y_pred) * X[j]
        
    # Add regularization to the weight gradient
    w_grad += 2*reg_param * w
    
    # Update Adagrad learning rate parameters
    lr_b += b_grad ** 2
    lr_w = lr_w+ w_grad ** 2
    
    # Update bias and weight using Adagrad learning rate
    b -= lr / np.sqrt(lr_b) * b_grad
    w = w- lr / ((lr_w)**0.5) * w_grad

# Print the final bias and weight values
print('Final bias:', b)
print('Final weights:', w)

# Calculate the predicted values using the final bias and weights
pred = np.dot(X, w) + b
print(pred)


# In[577]:


#RMSE: Above uses all training data to train the model
deviation=0
for j in range(X.shape[0]):
        y_pred = np.dot(w, X[j]) + b       
        deviation+= (y[j]-y_pred)**2
print(X.shape[0])
print(deviation)
RMSE= np.sqrt(deviation/X.shape[0])
print(RMSE)


# In[538]:





# In[498]:


#Reduced data 
reduced_data = method1.drop(['id','Output'],axis=1)
reduced_data = np.array(reduced_data)


# In[ ]:


#Performing Linear Regression with L2
y  = method1['Output']
y = np.array(y)
X= reduced_data

pred = np.zeros(X.shape[0])
# Normalize the features to have zero mean and unit variance
X = (X - np.mean(X, axis=0)) / np.std(X.astype(float), axis=0)
# Define initial bias, weight, learning rate, iteration, and regularization parameter
b = 0.0
w = np.zeros(X.shape[1])
lr = 0.01
iteration = 5000000
reg_param = 0.1

# Initialize Adagrad learning rate parameters
lr_b = 0.0
lr_w = np.zeros(X.shape[1])

# Perform stochastic gradient descent with Adagrad
for i in range(iteration):
    # Randomly select a single sample
    j = np.random.randint(X.shape[0])
    
    y_pred = np.dot(w, X[j]) + b
    b_grad = -2.0*(y[j] - y_pred) * 1
    w_grad = -2.0*(y[j] - y_pred) * X[j]
        
    # Add regularization to the weight gradient
    w_grad += 2*reg_param * w
    
    # Update Adagrad learning rate parameters
    lr_b += b_grad ** 2
    lr_w = lr_w+ w_grad ** 2
    
    # Update bias and weight using Adagrad learning rate
    b -= lr / np.sqrt(lr_b) * b_grad
    w = w- lr / ((lr_w)**0.5) * w_grad

# Print the final bias and weight values
print('Final bias:', b)
print('Final weights:', w)

# Calculate the predicted values using the final bias and weights
pred = np.dot(X, w) + b
print(pred)


# In[431]:


#Reduced data 2
import matplotlib.pyplot as plt

#Reduced data 
X = method1.drop(['id','Output'],axis=1)
X = np.array(X)


### Step 1: Standardize the Data along the Features
standardized_data = (X - np.mean(X, axis=0)) / np.std(X)

### Step 2: Calculate the Covariance Matrix
covariance_matrix = np.cov(standardized_data.astype(float), ddof = 0, rowvar = False)


### Step 3: Eigendecomposition on the Covariance Matrix
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)


### Step 4: Sort the Principal Components
# np.argsort can only provide lowest to highest; use [::-1] to reverse the list
order_of_importance = np.argsort(eigenvalues)[::-1] 

# utilize the sort order to sort eigenvalues and eigenvectors
sorted_eigenvalues = eigenvalues[order_of_importance]
sorted_eigenvectors = eigenvectors[:,order_of_importance] # sort the columns


### Step 5: Calculate the Explained Variance
# use sorted_eigenvalues to ensure the explained variances correspond to the eigenvectors
explained_variance = sorted_eigenvalues / np.sum(sorted_eigenvalues)


### Step 6: Reduce the Data via the Principal Components
k = 3 # select the number of principal components
reduced_data = np.matmul(standardized_data, sorted_eigenvectors[:,:k]) # transform the original data


# In[434]:


reduced_data


# In[435]:


#Performing Linear Regression
y  = method1['Output']
y = np.array(y)
X= reduced_data

pred = np.zeros(X.shape[0])
# Normalize the features to have zero mean and unit variance
X = (X - np.mean(X, axis=0)) / np.std(X.astype(float), axis=0)
# Define initial bias, weight, learning rate, iteration, and regularization parameter
b = 0.0
w = np.zeros(X.shape[1])
lr = 0.01
iteration = 3000000
reg_param = 0.1

# Initialize Adagrad learning rate parameters
lr_b = 0.0
lr_w = np.zeros(X.shape[1])

# Perform stochastic gradient descent with Adagrad
for i in range(iteration):
    # Randomly select a single sample
    j = np.random.randint(X.shape[0])
    
    y_pred = np.dot(w, X[j]) + b
    b_grad = -2.0*(y[j] - y_pred) * 1
    w_grad = -2.0*(y[j] - y_pred) * X[j]
    
    # Update Adagrad learning rate parameters
    lr_b += b_grad ** 2
    lr_w = lr_w+ w_grad ** 2
    
    # Update bias and weight using Adagrad learning rate
    b -= lr / np.sqrt(lr_b) * b_grad
    w = w- lr / ((lr_w)**0.5) * w_grad

# Print the final bias and weight values
print('Final bias:', b)
print('Final weights:', w)

# Calculate the predicted values using the final bias and weights
pred = np.dot(X, w) + b
print(pred)


# In[436]:


#RMSE: Above uses all training data to train the model
deviation=0
for j in range(X.shape[0]):
        y_pred = np.dot(w, X[j]) + b        
        deviation+= (y[j]-y_pred)**2
print(X.shape[0])
print(deviation)
RMSE= np.sqrt(deviation/X.shape[0])
print(RMSE)


# In[440]:


#Method 2: Using mean value as a variable

# Create a list of column names
columns = ['AMB_TEMPa', 'CH4a', 'COa', 'NMHCa', 'NOa', 'NO2a', 'NOXa', 'O3a', 'PM10a', 'PM2.5a','RAINFALLa', 'RHa', 'SO2a', 'THCa', 'WD_HRa', 'WIND_DIRECa', 'WIND_SPEEDa','WS_HRa','Output']
# Create an empty DataFrame with the specified columns
method2 = pd.DataFrame(columns=columns)
method2['Output']=pd.to_numeric(method2["Output"])

# insert Id column at beginning
method2.insert(0, 'id', range(0, 5652))


# iterate through each id
for i in range(5652):
    id_df = df.loc[df['id'] == i] # select rows with the iterated id value
    for index, row in id_df.iterrows():
        item_name = row['ItemName']
        mean =row['Mean']
        method2.loc[method2['id'] == i,item_name +('a')] = mean   
        # add PM2.5 value in Hour 10 of the iterated index to the new dataset column 'Output'
        if item_name=="PM2.5":
            pm25_value=row['Hour 10']
            method2.loc[method2['id'] == i, 'Output'] = pm25_value
        else:
            continue
   
        
# print the new dataset
print(method2)




# In[441]:


#Reduced data 
reduced_data = method2.drop(['id','Output'],axis=1)
reduced_data = np.array(reduced_data)


#Performing Linear Regression
y  = method2['Output']
y = np.array(y)
X= reduced_data

pred = np.zeros(X.shape[0])
# Normalize the features to have zero mean and unit variance
X = (X - np.mean(X, axis=0)) / np.std(X.astype(float), axis=0)
# Define initial bias, weight, learning rate, iteration, and regularization parameter
b = 0.0
w = np.zeros(X.shape[1])
lr = 0.01
iteration = 3000000
reg_param = 0.1

# Initialize Adagrad learning rate parameters
lr_b = 0.0
lr_w = np.zeros(X.shape[1])

# Perform stochastic gradient descent with Adagrad
for i in range(iteration):
    # Randomly select a single sample
    j = np.random.randint(X.shape[0])
    
    y_pred = np.dot(w, X[j]) + b
    b_grad = -2.0*(y[j] - y_pred) * 1
    w_grad = -2.0*(y[j] - y_pred) * X[j]
        
    # Add regularization to the weight gradient
    w_grad += 2*reg_param * w
    
    # Update Adagrad learning rate parameters
    lr_b += b_grad ** 2
    lr_w = lr_w+ w_grad ** 2
    
    # Update bias and weight using Adagrad learning rate
    b -= lr / np.sqrt(lr_b) * b_grad
    w = w- lr / ((lr_w)**0.5) * w_grad

# Print the final bias and weight values
print('Final bias:', b)
print('Final weights:', w)

# Calculate the predicted values using the final bias and weights
pred = np.dot(X, w) + b
print(pred)


# In[442]:


#RMSE: Above uses all training data to train the model
deviation=0
for j in range(X.shape[0]):
        y_pred = np.dot(w, X[j]) + b        
        deviation+= (y[j]-y_pred)**2
print(X.shape[0])
print(deviation)
RMSE= np.sqrt(deviation/X.shape[0])
print(RMSE)


# In[482]:


#Reduced data 3
method3 = method1[['Output','THCc','THCb','THCa',   'NOXc','NOXb','NOXa','SO2c','SO2b','SO2a','PM2.5a','PM2.5b','PM2.5c','AMB_TEMPa','AMB_TEMPb','AMB_TEMPc','RAINFALLa','RAINFALLb','RAINFALLc', 'WIND_SPEEDa','WIND_SPEEDb','WIND_SPEEDc','WIND_DIRECa','WIND_DIRECb','WIND_DIRECc','RHb','RHa','RHc']]


# In[495]:


#Reduced data 
reduced_data = method3.drop(['Output'],axis=1)
reduced_data = np.array(reduced_data)


#Performing Linear Regression
y  = method3['Output']
y = np.array(y)
X= reduced_data

pred = np.zeros(X.shape[0])
# Normalize the features to have zero mean and unit variance
X = (X - np.mean(X, axis=0)) / np.std(X.astype(float), axis=0)
# Define initial bias, weight, learning rate, iteration, and regularization parameter
b = 0.0
w = np.zeros(X.shape[1])
lr = 0.01
iteration = 5000000
reg_param = 0.1

# Initialize Adagrad learning rate parameters
lr_b = 0.0
lr_w = np.zeros(X.shape[1])

# Perform stochastic gradient descent with Adagrad
for i in range(iteration):
    # Randomly select a single sample
    j = np.random.randint(X.shape[0])
    
    y_pred = np.dot(w, X[j]) + b
    b_grad = -2.0*(y[j] - y_pred) * 1
    w_grad = -2.0*(y[j] - y_pred) * X[j]
        
    # Add regularization to the weight gradient
    #w_grad += 2*reg_param * w
    
    # Update Adagrad learning rate parameters
    lr_b += b_grad ** 2
    lr_w = lr_w+ w_grad ** 2
    
    # Update bias and weight using Adagrad learning rate
    b -= lr / np.sqrt(lr_b) * b_grad
    w = w- lr / ((lr_w)**0.5) * w_grad

# Print the final bias and weight values
print('Final bias:', b)
print('Final weights:', w)

# Calculate the predicted values using the final bias and weights
pred = np.dot(X, w) + b
print(pred)


# In[496]:


#RMSE: Above uses all training data to train the model
deviation=0
for j in range(X.shape[0]):
        y_pred = np.dot(w, X[j]) + b        
        deviation+= (y[j]-y_pred)**2
print(X.shape[0])
print(deviation)
RMSE= np.sqrt(deviation/X.shape[0])
print(RMSE)


# In[485]:


#Reduced data 4
method4 = method1[['Output','PM2.5a','PM2.5b','PM2.5c']]


# In[486]:


#Reduced data 
reduced_data = method4.drop(['Output'],axis=1)
reduced_data = np.array(reduced_data)


#Performing Linear Regression
y  = method4['Output']
y = np.array(y)
X= reduced_data

pred = np.zeros(X.shape[0])
# Normalize the features to have zero mean and unit variance
X = (X - np.mean(X, axis=0)) / np.std(X.astype(float), axis=0)
# Define initial bias, weight, learning rate, iteration, and regularization parameter
b = 0.0
w = np.zeros(X.shape[1])
lr = 0.01
iteration = 5000000
reg_param = 0.1

# Initialize Adagrad learning rate parameters
lr_b = 0.0
lr_w = np.zeros(X.shape[1])

# Perform stochastic gradient descent with Adagrad
for i in range(iteration):
    # Randomly select a single sample
    j = np.random.randint(X.shape[0])
    
    y_pred = np.dot(w, X[j]) + b
    b_grad = -2.0*(y[j] - y_pred) * 1
    w_grad = -2.0*(y[j] - y_pred) * X[j]
        
    # Add regularization to the weight gradient
    #w_grad += 2*reg_param * w
    
    # Update Adagrad learning rate parameters
    lr_b += b_grad ** 2
    lr_w = lr_w+ w_grad ** 2
    
    # Update bias and weight using Adagrad learning rate
    b -= lr / np.sqrt(lr_b) * b_grad
    w = w- lr / ((lr_w)**0.5) * w_grad

# Print the final bias and weight values
print('Final bias:', b)
print('Final weights:', w)

# Calculate the predicted values using the final bias and weights
pred = np.dot(X, w) + b
print(pred)


# In[487]:


#RMSE: Above uses all training data to train the model
deviation=0
for j in range(X.shape[0]):
        y_pred = np.dot(w, X[j]) + b        
        deviation+= (y[j]-y_pred)**2
print(X.shape[0])
print(deviation)
RMSE= np.sqrt(deviation/X.shape[0])
print(RMSE)


# In[507]:


#Estimate NA values for test data set as well

#Processing test data



# Read CSV file
final = pd.read_csv(r"C:\Users\Phoebe Tan\OneDrive - National University of Singapore\Desktop\Taiwan\Data Mining\test_X.csv", header=None)

# Create a list of new headers
new_headers = ["Index","ItemName", "Hour 1", "Hour 2", "Hour 3", "Hour 4", "Hour 5", "Hour 6", "Hour 7", "Hour 8", "Hour 9"]

# Insert the new headers as the first row of the dataframe
final.columns = new_headers + [str(i) for i in range(1, len(final.columns) - len(new_headers) + 1)]
final.columns = [col.strip() for col in final.columns]
# Display the modified dataframe
print(final)

   
print(final.loc[final['Index']=='index_0'])


# In[508]:


# Replace index_i with i
final['Index'] = final['Index'].replace({'index_': ''}, regex=True)

# convert the column Index from character to integer
final['Index'] = final['Index'].astype(int)


# In[509]:


# replace the invalid values with NaN
final = final.replace(['AMB_TEMP'], 'TEMP', regex=True)
final = final.replace(['RAINFALL'], 'RF', regex=True)
final = final.replace(['NOx'], 'NO3', regex=True)
final = final.replace(['#\s*', '\*\s*', 'x\s*', 'A\s*'], pd.NA, regex=True)
final = final.replace(['TEMP'], 'AMB_TEMP', regex=True)
final = final.replace(['NO3'], 'NOX', regex=True)
final = final.replace(['RF'], 'RAINFALL', regex=True)


# In[510]:


rows_with_missing_values = final[final.isnull().any(axis=1)]
print(rows_with_missing_values)


# In[511]:


final.iloc[:, 2:] = final.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')
final


# In[512]:


# Fill in missing values in each row with the row mean
for i, row in final.iterrows():
    row_values = row.values[2:]  # extract values from column 3 onwards
    non_na_values = [x for x in row_values if not pd.isna(x)]
    if len(non_na_values) > 0:
        row_mean = np.mean(non_na_values)
        row.fillna(row_mean, inplace=True)
        final.loc[i,:]= row
        
    else:
        print(i)


# In[513]:


final.iloc[66-18,:]


# In[514]:


p=66

hour1= ( final.iloc[p-18,2] + final.iloc[p+18,2] )/2
hour2= ( final.iloc[p-18,3] + final.iloc[p+18,3] )/2
hour3= ( final.iloc[p-18,4] + final.iloc[p+18,4] )/2
hour4= ( final.iloc[p-18,5] + final.iloc[p+18,5] )/2
hour5= ( final.iloc[p-18,6] + final.iloc[p+18,6] )/2
hour6= ( final.iloc[p-18,7] + final.iloc[p+18,7] )/2
hour7= ( final.iloc[p-18,8] + final.iloc[p+18,8] )/2
hour8= ( final.iloc[p-18,9] + final.iloc[p+18,9] )/2
hour9= ( final.iloc[p-18,10] + final.iloc[p+18,10] )/2



final.at[p, 'Hour 1'] = hour1
final.at[p, 'Hour 2'] = hour2
final.at[p, 'Hour 3'] = hour3zW
final.at[p, 'Hour 4'] = hour4
final.at[p, 'Hour 5'] = hour5
final.at[p, 'Hour 6'] = hour6
final.at[p, 'Hour 7'] = hour7
final.at[p, 'Hour 8'] = hour8
final.at[p, 'Hour 9'] = hour9


# In[515]:


p=138

hour1= ( final.iloc[p-18,2] + final.iloc[p+18,2] )/2
hour2= ( final.iloc[p-18,3] + final.iloc[p+18,3] )/2
hour3= ( final.iloc[p-18,4] + final.iloc[p+18,4] )/2
hour4= ( final.iloc[p-18,5] + final.iloc[p+18,5] )/2
hour5= ( final.iloc[p-18,6] + final.iloc[p+18,6] )/2
hour6= ( final.iloc[p-18,7] + final.iloc[p+18,7] )/2
hour7= ( final.iloc[p-18,8] + final.iloc[p+18,8] )/2
hour8= ( final.iloc[p-18,9] + final.iloc[p+18,9] )/2
hour9= ( final.iloc[p-18,10] + final.iloc[p+18,10] )/2



final.at[p, 'Hour 1'] = hour1
final.at[p, 'Hour 2'] = hour2
final.at[p, 'Hour 3'] = hour3
final.at[p, 'Hour 4'] = hour4
final.at[p, 'Hour 5'] = hour5
final.at[p, 'Hour 6'] = hour6
final.at[p, 'Hour 7'] = hour7
final.at[p, 'Hour 8'] = hour8
final.at[p, 'Hour 9'] = hour9


# In[516]:


final.iloc[p,:]


# In[517]:


p=2352

hour1= ( final.iloc[p-18,2] + final.iloc[p+18,2] )/2
hour2= ( final.iloc[p-18,3] + final.iloc[p+18,3] )/2
hour3= ( final.iloc[p-18,4] + final.iloc[p+18,4] )/2
hour4= ( final.iloc[p-18,5] + final.iloc[p+18,5] )/2
hour5= ( final.iloc[p-18,6] + final.iloc[p+18,6] )/2
hour6= ( final.iloc[p-18,7] + final.iloc[p+18,7] )/2
hour7= ( final.iloc[p-18,8] + final.iloc[p+18,8] )/2
hour8= ( final.iloc[p-18,9] + final.iloc[p+18,9] )/2
hour9= ( final.iloc[p-18,10] + final.iloc[p+18,10] )/2



final.at[p, 'Hour 1'] = hour1
final.at[p, 'Hour 2'] = hour2
final.at[p, 'Hour 3'] = hour3
final.at[p, 'Hour 4'] = hour4
final.at[p, 'Hour 5'] = hour5
final.at[p, 'Hour 6'] = hour6
final.at[p, 'Hour 7'] = hour7
final.at[p, 'Hour 8'] = hour8
final.at[p, 'Hour 9'] = hour9


# In[518]:


final.iloc[p,:]


# In[519]:


rows_with_missing_values = final[final.isnull().any(axis=1)]
print(rows_with_missing_values) #All NA Values have been filled


# In[539]:


final


# In[542]:


# calculate the means for each subset of columns
final["Mean1"] = final.iloc[:, 2:4].mean(axis=1)
final["Mean2"] = final.iloc[:, 5:7].mean(axis=1)
final["Mean3"] = final.iloc[:, 8:10].mean(axis=1)


# In[543]:


final


# In[550]:


# Feature selection: Aggregation (Method 1- Take 2) Take mean of the 3hours data as a feature for each variable to predict 10 hours)
final.columns = [col.strip() for col in final.columns]
final['ItemName']=final['ItemName'].str.strip()
# Create a list of column names
columns = ['AMB_TEMPa', 'CH4a', 'COa', 'NMHCa', 'NOa', 'NO2a', 'NOXa', 'O3a', 'PM10a', 'PM2.5a','RAINFALLa', 'RHa', 'SO2a', 'THCa', 'WD_HRa', 'WIND_DIRECa', 'WIND_SPEEDa','WS_HRa','AMB_TEMPb', 'CH4b', 'COb', 'NMHCb', 'NOb', 'NO2b', 'NOXb', 'O3b', 'PM10b', 'PM2.5b','RAINFALLb', 'RHb', 'SO2b', 'THCb', 'WD_HRb', 'WIND_DIRECb', 'WIND_SPEEDb','WS_HRb','AMB_TEMPc', 'CH4c', 'COc', 'NMHCc', 'NOc', 'NO2c', 'NOXc', 'O3c', 'PM10c', 'PM2.5c','RAINFALLc', 'RHc', 'SO2c', 'THCc', 'WD_HRc', 'WIND_DIRECc', 'WIND_SPEEDc','WS_HRc']
# Create an empty DataFrame with the specified columns
final1 = pd.DataFrame(columns=columns)


# insert Id column at beginning
final1.insert(0, 'id', range(0, 244))


# iterate through each id
for i in range(244):
    id_df = final.loc[final['Index'] == i] # select rows with the iterated id value
    for index, row in id_df.iterrows():
        item_name = row['ItemName']
        mean1 =row['Mean1']
        mean2 =row['Mean2']
        mean3=row['Mean3']
        final1.loc[final1['id'] == i,item_name +('a')] = mean1 
        final1.loc[final1['id'] == i,item_name+('b')] = mean2
        final1.loc[final1['id'] == i,item_name+('c')] = mean3     

        
# print the new dataset
print(final1)


# In[565]:


X = final1.drop(['id'],axis=1)
X = np.array(X)
# Normalize the features to have zero mean and unit variance
X = (X - np.mean(X, axis=0)) / np.std(X.astype(float), axis=0)


# In[566]:


X


# In[567]:


#Initialize y predicted values
predicted= np.zeros(X.shape[0])


for j in range(X.shape[0]):
    y_pred = np.dot(w, X[j]) + b
    predicted[j]=y_pred
 


# Print the final bias and weight values
print('Final bias:', b)
print('Final weights:', w)


# In[568]:


print(predicted)


# In[570]:


import csv
# Transpose the array
predicted = np.transpose(predicted)



# Open the CSV file in write mode
with open('outputfinal3.csv', mode='w', newline='') as file:

    # Create a CSV writer object
    writer = csv.writer(file)

    # Write the array as a row
    writer.writerow(predicted)


# In[571]:


# Read the row CSV file
df_row = pd.read_csv('outputfinal3.csv')

# Transpose the DataFrame
df_col = df_row.T

# Write the column CSV file
df_col.to_csv('finaloutput3.csv', header=False)


# In[ ]:


#WITH REMOVED NA VALUES, best method


import pandas as pd
import numpy as np

# read in the data
data = pd.read_csv(r"C:\Users\Phoebe Tan\OneDrive - National University of Singapore\Desktop\Taiwan\Data Mining\train.csv", sep=',', header=0, skipinitialspace=True)
data.columns = [col.strip() for col in data.columns]
data['ItemName']=data['ItemName'].str.strip()

# Load the data from CSV
df=data

# Convert the 'Date' column to a datetime format
df['Date'] = pd.to_datetime(df['Date'])


# In[ ]:


# replace the invalid values with NaN
df = df.replace(['AMB_TEMP'], 'TEMP', regex=True)
df = df.replace(['RAINFALL'], 'RF', regex=True)
df = df.replace(['NOx'], 'NO3', regex=True)
df = df.replace(['#\s*', '\*\s*', 'x\s*', 'A\s*'], pd.NA, regex=True)
df = df.replace(['TEMP'], 'AMB_TEMP', regex=True)
df = df.replace(['NO3'], 'NOX', regex=True)
df = df.replace(['RF'], 'RAINFALL', regex=True)


# In[ ]:


df[['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23']]= df[['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23']].apply(pd.to_numeric)


# In[ ]:


# Extract the month from the 'date' column
df['Month'] = df['Date'].dt.month
print(df)


# In[ ]:


# Find the row indices with at least one NaN value
nan_rows = df.isna().any(axis=1)

# Print the number of rows with NA values
print('Number of rows with NA values:',sum(nan_rows==True))


# In[ ]:


#Separate by month

month1 = df[df['Month'] == 1]
month2 = df[df['Month'] == 2]
month3 = df[df['Month'] == 3]
month4 = df[df['Month'] == 4]
month5 = df[df['Month'] == 5]
month6 = df[df['Month'] == 6]
month7 = df[df['Month'] == 7]
month8 = df[df['Month'] == 8]
month9 = df[df['Month'] == 9]
month10 = df[df['Month'] ==10]
month11 = df[df['Month'] == 11]
month12 = df[df['Month'] == 12]


# In[ ]:


# iterate over each unique item in the original dataframe
df=month1.drop(["Month","Location","Date"], axis=1)
data_dict={}
for item in df['ItemName'].unique():
    # get the rows for the current item
    item_rows = df[df['ItemName'] == item]
    tempvalues=[]
    data_dict[item]=tempvalues
    for i, row in item_rows.iterrows():
        for col in item_rows.columns[1:]:
            tempvalues.append(row[col])
    
# Define the size of the sliding window
window_size = 10

# Initialize an empty list to hold the results
result_list = []

# Iterate over each key-value pair in the dictionary
for item_name, values in data_dict.items():
    # Iterate over each starting index for the window
    for start_idx in range(len(values) - window_size + 1):
        # Get the values for the current window
        window_values = values[start_idx : start_idx + window_size]
        
        # Append the results to the result list
        result_list.append([item_name] + window_values)

# Convert the results into a pandas DataFrame
column_names = ['ItemName'] + [f'Hour {i+1}' for i in range(window_size)]
mth1 = pd.DataFrame(result_list, columns=column_names)

# Print the resulting DataFrame
print(mth1)



mth1.to_csv("month1.00.csv", index=False)


# In[ ]:


# Find the row indices with at least one NaN value
nan_rows = mth1.isna().any(axis=1)

# Print the number of rows with NA values
print('Number of rows with NA values:',sum(nan_rows==True))


# In[ ]:


# iterate over each unique item in the original dataframe
df=month2.drop(["Month","Location","Date"], axis=1)
data_dict={}
for item in df['ItemName'].unique():
    # get the rows for the current item
    item_rows = df[df['ItemName'] == item]
    tempvalues=[]
    data_dict[item]=tempvalues
    for i, row in item_rows.iterrows():
        for col in item_rows.columns[1:]:
            tempvalues.append(row[col])
    
# Define the size of the sliding window
window_size = 10

# Initialize an empty list to hold the results
result_list = []

# Iterate over each key-value pair in the dictionary
for item_name, values in data_dict.items():
    # Iterate over each starting index for the window
    for start_idx in range(len(values) - window_size + 1):
        # Get the values for the current window
        window_values = values[start_idx : start_idx + window_size]
        
        # Append the results to the result list
        result_list.append([item_name] + window_values)

# Convert the results into a pandas DataFrame
column_names = ['ItemName'] + [f'Hour {i+1}' for i in range(window_size)]
mth2 = pd.DataFrame(result_list, columns=column_names)

# Print the resulting DataFrame
print(mth2)
#result_df.to_csv("month1.00.csv", index=False)


# In[ ]:


# iterate over each unique item in the original dataframe
df=month3.drop(["Month","Location","Date"], axis=1)
data_dict={}
for item in df['ItemName'].unique():
    # get the rows for the current item
    item_rows = df[df['ItemName'] == item]
    tempvalues=[]
    data_dict[item]=tempvalues
    for i, row in item_rows.iterrows():
        for col in item_rows.columns[1:]:
            tempvalues.append(row[col])
    
# Define the size of the sliding window
window_size = 10

# Initialize an empty list to hold the results
result_list = []

# Iterate over each key-value pair in the dictionary
for item_name, values in data_dict.items():
    # Iterate over each starting index for the window
    for start_idx in range(len(values) - window_size + 1):
        # Get the values for the current window
        window_values = values[start_idx : start_idx + window_size]
        
        # Append the results to the result list
        result_list.append([item_name] + window_values)

# Convert the results into a pandas DataFrame
column_names = ['ItemName'] + [f'Hour {i+1}' for i in range(window_size)]
mth3 = pd.DataFrame(result_list, columns=column_names)

# Print the resulting DataFrame
print(mth3)
#result_df.to_csv("month1.00.csv", index=False)


# In[ ]:


# iterate over each unique item in the original dataframe
df=month4.drop(["Month","Location","Date"], axis=1)
data_dict={}
for item in df['ItemName'].unique():
    # get the rows for the current item
    item_rows = df[df['ItemName'] == item]
    tempvalues=[]
    data_dict[item]=tempvalues
    for i, row in item_rows.iterrows():
        for col in item_rows.columns[1:]:
            tempvalues.append(row[col])
    
# Define the size of the sliding window
window_size = 10

# Initialize an empty list to hold the results
result_list = []

# Iterate over each key-value pair in the dictionary
for item_name, values in data_dict.items():
    # Iterate over each starting index for the window
    for start_idx in range(len(values) - window_size + 1):
        # Get the values for the current window
        window_values = values[start_idx : start_idx + window_size]
        
        # Append the results to the result list
        result_list.append([item_name] + window_values)

# Convert the results into a pandas DataFrame
column_names = ['ItemName'] + [f'Hour {i+1}' for i in range(window_size)]
mth4 = pd.DataFrame(result_list, columns=column_names)

# Print the resulting DataFrame
print(mth4)
#result_df.to_csv("month1.00.csv", index=False)


# In[ ]:


# iterate over each unique item in the original dataframe
df=month5.drop(["Month","Location","Date"], axis=1)
data_dict={}
for item in df['ItemName'].unique():
    # get the rows for the current item
    item_rows = df[df['ItemName'] == item]
    tempvalues=[]
    data_dict[item]=tempvalues
    for i, row in item_rows.iterrows():
        for col in item_rows.columns[1:]:
            tempvalues.append(row[col])
    
# Define the size of the sliding window
window_size = 10

# Initialize an empty list to hold the results
result_list = []

# Iterate over each key-value pair in the dictionary
for item_name, values in data_dict.items():
    # Iterate over each starting index for the window
    for start_idx in range(len(values) - window_size + 1):
        # Get the values for the current window
        window_values = values[start_idx : start_idx + window_size]
        
        # Append the results to the result list
        result_list.append([item_name] + window_values)

# Convert the results into a pandas DataFrame
column_names = ['ItemName'] + [f'Hour {i+1}' for i in range(window_size)]
mth5 = pd.DataFrame(result_list, columns=column_names)

# Print the resulting DataFrame
print(mth5)
#result_df.to_csv("month1.00.csv", index=False)


# In[ ]:


# iterate over each unique item in the original dataframe
df=month6.drop(["Month","Location","Date"], axis=1)
data_dict={}
for item in df['ItemName'].unique():
    # get the rows for the current item
    item_rows = df[df['ItemName'] == item]
    tempvalues=[]
    data_dict[item]=tempvalues
    for i, row in item_rows.iterrows():
        for col in item_rows.columns[1:]:
            tempvalues.append(row[col])
    
# Define the size of the sliding window
window_size = 10

# Initialize an empty list to hold the results
result_list = []

# Iterate over each key-value pair in the dictionary
for item_name, values in data_dict.items():
    # Iterate over each starting index for the window
    for start_idx in range(len(values) - window_size + 1):
        # Get the values for the current window
        window_values = values[start_idx : start_idx + window_size]
        
        # Append the results to the result list
        result_list.append([item_name] + window_values)

# Convert the results into a pandas DataFrame
column_names = ['ItemName'] + [f'Hour {i+1}' for i in range(window_size)]
mth6 = pd.DataFrame(result_list, columns=column_names)

# Print the resulting DataFrame
print(mth6)
#result_df.to_csv("month1.00.csv", index=False)


# In[ ]:


# iterate over each unique item in the original dataframe
df=month7.drop(["Month","Location","Date"], axis=1)
data_dict={}
for item in df['ItemName'].unique():
    # get the rows for the current item
    item_rows = df[df['ItemName'] == item]
    tempvalues=[]
    data_dict[item]=tempvalues
    for i, row in item_rows.iterrows():
        for col in item_rows.columns[1:]:
            tempvalues.append(row[col])
    
# Define the size of the sliding window
window_size = 10

# Initialize an empty list to hold the results
result_list = []

# Iterate over each key-value pair in the dictionary
for item_name, values in data_dict.items():
    # Iterate over each starting index for the window
    for start_idx in range(len(values) - window_size + 1):
        # Get the values for the current window
        window_values = values[start_idx : start_idx + window_size]
        
        # Append the results to the result list
        result_list.append([item_name] + window_values)

# Convert the results into a pandas DataFrame
column_names = ['ItemName'] + [f'Hour {i+1}' for i in range(window_size)]
mth7 = pd.DataFrame(result_list, columns=column_names)

# Print the resulting DataFrame
print(mth7)
#result_df.to_csv("month1.00.csv", index=False)


# In[ ]:


# iterate over each unique item in the original dataframe
df=month8.drop(["Month","Location","Date"], axis=1)
data_dict={}
for item in df['ItemName'].unique():
    # get the rows for the current item
    item_rows = df[df['ItemName'] == item]
    tempvalues=[]
    data_dict[item]=tempvalues
    for i, row in item_rows.iterrows():
        for col in item_rows.columns[1:]:
            tempvalues.append(row[col])
    
# Define the size of the sliding window
window_size = 10

# Initialize an empty list to hold the results
result_list = []

# Iterate over each key-value pair in the dictionary
for item_name, values in data_dict.items():
    # Iterate over each starting index for the window
    for start_idx in range(len(values) - window_size + 1):
        # Get the values for the current window
        window_values = values[start_idx : start_idx + window_size]
        
        # Append the results to the result list
        result_list.append([item_name] + window_values)

# Convert the results into a pandas DataFrame
column_names = ['ItemName'] + [f'Hour {i+1}' for i in range(window_size)]
mth8 = pd.DataFrame(result_list, columns=column_names)

# Print the resulting DataFrame
print(mth8)
#result_df.to_csv("month1.00.csv", index=False)


# In[ ]:


# iterate over each unique item in the original dataframe
df=month9.drop(["Month","Location","Date"], axis=1)
data_dict={}
for item in df['ItemName'].unique():
    # get the rows for the current item
    item_rows = df[df['ItemName'] == item]
    tempvalues=[]
    data_dict[item]=tempvalues
    for i, row in item_rows.iterrows():
        for col in item_rows.columns[1:]:
            tempvalues.append(row[col])
    
# Define the size of the sliding window
window_size = 10

# Initialize an empty list to hold the results
result_list = []

# Iterate over each key-value pair in the dictionary
for item_name, values in data_dict.items():
    # Iterate over each starting index for the window
    for start_idx in range(len(values) - window_size + 1):
        # Get the values for the current window
        window_values = values[start_idx : start_idx + window_size]
        
        # Append the results to the result list
        result_list.append([item_name] + window_values)

# Convert the results into a pandas DataFrame
column_names = ['ItemName'] + [f'Hour {i+1}' for i in range(window_size)]
mth9 = pd.DataFrame(result_list, columns=column_names)

# Print the resulting DataFrame
print(mth9)
#result_df.to_csv("month1.00.csv", index=False)


# In[ ]:


# iterate over each unique item in the original dataframe
df=month10.drop(["Month","Location","Date"], axis=1)
data_dict={}
for item in df['ItemName'].unique():
    # get the rows for the current item
    item_rows = df[df['ItemName'] == item]
    tempvalues=[]
    data_dict[item]=tempvalues
    for i, row in item_rows.iterrows():
        for col in item_rows.columns[1:]:
            tempvalues.append(row[col])
    
# Define the size of the sliding window
window_size = 10

# Initialize an empty list to hold the results
result_list = []

# Iterate over each key-value pair in the dictionary
for item_name, values in data_dict.items():
    # Iterate over each starting index for the window
    for start_idx in range(len(values) - window_size + 1):
        # Get the values for the current window
        window_values = values[start_idx : start_idx + window_size]
        
        # Append the results to the result list
        result_list.append([item_name] + window_values)

# Convert the results into a pandas DataFrame
column_names = ['ItemName'] + [f'Hour {i+1}' for i in range(window_size)]
mth10 = pd.DataFrame(result_list, columns=column_names)

# Print the resulting DataFrame
print(mth10)
#result_df.to_csv("month1.00.csv", index=False)


# In[ ]:


# iterate over each unique item in the original dataframe
df=month11.drop(["Month","Location","Date"], axis=1)
data_dict={}
for item in df['ItemName'].unique():
    # get the rows for the current item
    item_rows = df[df['ItemName'] == item]
    tempvalues=[]
    data_dict[item]=tempvalues
    for i, row in item_rows.iterrows():
        for col in item_rows.columns[1:]:
            tempvalues.append(row[col])
    
# Define the size of the sliding window
window_size = 10

# Initialize an empty list to hold the results
result_list = []

# Iterate over each key-value pair in the dictionary
for item_name, values in data_dict.items():
    # Iterate over each starting index for the window
    for start_idx in range(len(values) - window_size + 1):
        # Get the values for the current window
        window_values = values[start_idx : start_idx + window_size]
        
        # Append the results to the result list
        result_list.append([item_name] + window_values)

# Convert the results into a pandas DataFrame
column_names = ['ItemName'] + [f'Hour {i+1}' for i in range(window_size)]
mth11 = pd.DataFrame(result_list, columns=column_names)

# Print the resulting DataFrame
print(mth11)
#result_df.to_csv("month1.00.csv", index=False)


# In[ ]:


# iterate over each unique item in the original dataframe
df=month12.drop(["Month","Location","Date"], axis=1)
data_dict={}
for item in df['ItemName'].unique():
    # get the rows for the current item
    item_rows = df[df['ItemName'] == item]
    tempvalues=[]
    data_dict[item]=tempvalues
    for i, row in item_rows.iterrows():
        for col in item_rows.columns[1:]:
            tempvalues.append(row[col])
    
# Define the size of the sliding window
window_size = 10

# Initialize an empty list to hold the results
result_list = []

# Iterate over each key-value pair in the dictionary
for item_name, values in data_dict.items():
    # Iterate over each starting index for the window
    for start_idx in range(len(values) - window_size + 1):
        # Get the values for the current window
        window_values = values[start_idx : start_idx + window_size]
        
        # Append the results to the result list
        result_list.append([item_name] + window_values)

# Convert the results into a pandas DataFrame
column_names = ['ItemName'] + [f'Hour {i+1}' for i in range(window_size)]
mth12 = pd.DataFrame(result_list, columns=column_names)

# Print the resulting DataFrame
print(mth12)
#result_df.to_csv("month1.00.csv", index=False)


# In[ ]:


# Concatenate the DataFrames vertically
df= pd.concat([mth1,mth2,mth3,mth4,mth5,mth6,mth7,mth8,mth9,mth10,mth11,mth12])

# Print the resulting DataFrame
print(df)


# In[ ]:


# calculate the mean and median of the Hour 1 to 9 columns
df["Mean"] = df.iloc[:, 1:9].mean(axis=1)
df["Median"] = df.iloc[:, 1:9].median(axis=1)

# calculate the means for each subset of columns
df["Mean1"] = df.iloc[:, 1:3].mean(axis=1)
df["Mean2"] = df.iloc[:, 4:6].mean(axis=1)
df["Mean3"] = df.iloc[:, 7:9].mean(axis=1)

df


# In[ ]:


# Label each group with an ID from 0 to the length of the group
df['id'] = df.groupby('ItemName').cumcount()

# Print the resulting DataFrame
print(df)


# In[ ]:


# Find the id values of rows with at least one NA value
ids_to_drop = df.loc[df.isna().any(axis=1), 'id'].unique()

# Drop all rows with those id values
df = df[~df['id'].isin(ids_to_drop)]


# In[ ]:


# Find the row indices with at least one NaN value
nan_rows = df.isna().any(axis=1)

# Print the number of rows with NA values
print('Number of rows with NA values:',sum(nan_rows==True))


# In[ ]:


# Feature selection: Aggregation (Method 1- Take 2) Take mean of the 3hours data as a feature for each variable to predict 10 hours)

# Create a list of column names
columns = ['AMB_TEMPa', 'CH4a', 'COa', 'NMHCa', 'NOa', 'NO2a', 'NOXa', 'O3a', 'PM10a', 'PM2.5a','RAINFALLa', 'RHa', 'SO2a', 'THCa', 'WD_HRa', 'WIND_DIRECa', 'WIND_SPEEDa','WS_HRa','AMB_TEMPb', 'CH4b', 'COb', 'NMHCb', 'NOb', 'NO2b', 'NOXb', 'O3b', 'PM10b', 'PM2.5b','RAINFALLb', 'RHb', 'SO2b', 'THCb', 'WD_HRb', 'WIND_DIRECb', 'WIND_SPEEDb','WS_HRb','AMB_TEMPc', 'CH4c', 'COc', 'NMHCc', 'NOc', 'NO2c', 'NOXc', 'O3c', 'PM10c', 'PM2.5c','RAINFALLc', 'RHc', 'SO2c', 'THCc', 'WD_HRc', 'WIND_DIRECc', 'WIND_SPEEDc','WS_HRc','Output']
# Create an empty DataFrame with the specified columns
method1 = pd.DataFrame(columns=columns)
method1['Output']=pd.to_numeric(method1["Output"])

# insert Id column at beginning
method1.insert(0, 'id', range(0, 5652))


# iterate through each id
for i in range(5652):
    id_df = df.loc[df['id'] == i] # select rows with the iterated id value
    for index, row in id_df.iterrows():
        item_name = row['ItemName']
        mean1 =row['Mean1']
        mean2 =row['Mean2']
        mean3=row['Mean3']
        method1.loc[method1['id'] == i,item_name +('a')] = mean1 
        method1.loc[method1['id'] == i,item_name+('b')] = mean2
        method1.loc[method1['id'] == i,item_name+('c')] = mean3     
        # add PM2.5 value in Hour 10 of the iterated index to the new dataset column 'Output'
        if item_name=="PM2.5":
            pm25_value=row['Hour 10']
            method1.loc[method1['id'] == i, 'Output'] = pm25_value
        else:
            continue
   
        
# print the new dataset
print(method1)


# In[ ]:


method1 = method1.dropna()


# In[ ]:


#Reduced data 
reduced_data = method1.drop(['id','Output'],axis=1)
reduced_data = np.array(reduced_data)


# In[ ]:


#Performing Linear Regression
y  = method1['Output']
y = np.array(y)
X= reduced_data

pred = np.zeros(X.shape[0])
# Normalize the features to have zero mean and unit variance
X = (X - np.mean(X, axis=0)) / np.std(X.astype(float), axis=0)
# Define initial bias, weight, learning rate, iteration, and regularization parameter
b = 0.0
w = np.zeros(X.shape[1])
lr = 0.03
iteration = 8000000
reg_param = 0.1

# Initialize Adagrad learning rate parameters
lr_b = 0.0
lr_w = np.zeros(X.shape[1])

# Perform stochastic gradient descent with Adagrad
for i in range(iteration):
    # Randomly select a single sample
    j = np.random.randint(X.shape[0])
    
    y_pred = np.dot(w, X[j]) + b
    b_grad = -2.0*(y[j] - y_pred) * 1
    w_grad = -2.0*(y[j] - y_pred) * X[j]
        
    # Add regularization to the weight gradient
    w_grad += 2*reg_param * w
    
    # Update Adagrad learning rate parameters
    lr_b += b_grad ** 2
    lr_w = lr_w+ w_grad ** 2
    
    # Update bias and weight using Adagrad learning rate
    b -= lr / np.sqrt(lr_b) * b_grad
    w = w- lr / ((lr_w)**0.5) * w_grad

# Print the final bias and weight values
print('Final bias:', b)
print('Final weights:', w)

# Calculate the predicted values using the final bias and weights
pred = np.dot(X, w) + b
print(pred)


# In[ ]:


#RMSE: Above uses all training data to train the model
deviation=0
for j in range(X.shape[0]):
        y_pred = np.dot(w, X[j]) + b      
        deviation+= (y[j]-y_pred)**2
print(X.shape[0])
print(deviation)
RMSE= np.sqrt(deviation/X.shape[0])
print(RMSE)


# In[ ]:


#Remove NA values for test data set as well

#Processing test data



# Read CSV file
final = pd.read_csv(r"C:\Users\Phoebe Tan\OneDrive - National University of Singapore\Desktop\Taiwan\Data Mining\test_X.csv", header=None)

# Create a list of new headers
new_headers = ["Index","ItemName", "Hour 1", "Hour 2", "Hour 3", "Hour 4", "Hour 5", "Hour 6", "Hour 7", "Hour 8", "Hour 9"]

# Insert the new headers as the first row of the dataframe
final.columns = new_headers + [str(i) for i in range(1, len(final.columns) - len(new_headers) + 1)]
final.columns = [col.strip() for col in final.columns]
# Display the modified dataframe
print(final)

   
print(final.loc[final['Index']=='index_0'])


# In[ ]:


# Replace index_i with i
final['Index'] = final['Index'].replace({'index_': ''}, regex=True)

# convert the column Index from character to integer
final['Index'] = final['Index'].astype(int)


# In[ ]:


# replace the invalid values with NaN
final = final.replace(['AMB_TEMP'], 'TEMP', regex=True)
final = final.replace(['RAINFALL'], 'RF', regex=True)
final = final.replace(['NOx'], 'NO3', regex=True)
final = final.replace(['#\s*', '\*\s*', 'x\s*', 'A\s*'], pd.NA, regex=True)
final = final.replace(['TEMP'], 'AMB_TEMP', regex=True)
final = final.replace(['NO3'], 'NOX', regex=True)
final = final.replace(['RF'], 'RAINFALL', regex=True)


# In[ ]:


rows_with_missing_values = final[final.isnull().any(axis=1)]
print(rows_with_missing_values)


# In[ ]:


final.iloc[:, 2:] = final.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')
final


# In[ ]:


# Fill in missing values in each row with the row mean
for i, row in final.iterrows():
    row_values = row.values[2:]  # extract values from column 3 onwards
    non_na_values = [x for x in row_values if not pd.isna(x)]
    if len(non_na_values) > 0:
        row_mean = np.mean(non_na_values)
        row.fillna(row_mean, inplace=True)
        final.loc[i,:]= row
        
    else:
        print(i)


# In[ ]:


p=66

hour1= ( final.iloc[p-18,2] + final.iloc[p+18,2] )/2
hour2= ( final.iloc[p-18,3] + final.iloc[p+18,3] )/2
hour3= ( final.iloc[p-18,4] + final.iloc[p+18,4] )/2
hour4= ( final.iloc[p-18,5] + final.iloc[p+18,5] )/2
hour5= ( final.iloc[p-18,6] + final.iloc[p+18,6] )/2
hour6= ( final.iloc[p-18,7] + final.iloc[p+18,7] )/2
hour7= ( final.iloc[p-18,8] + final.iloc[p+18,8] )/2
hour8= ( final.iloc[p-18,9] + final.iloc[p+18,9] )/2
hour9= ( final.iloc[p-18,10] + final.iloc[p+18,10] )/2



final.at[p, 'Hour 1'] = hour1
final.at[p, 'Hour 2'] = hour2
final.at[p, 'Hour 3'] = hour3
final.at[p, 'Hour 4'] = hour4
final.at[p, 'Hour 5'] = hour5
final.at[p, 'Hour 6'] = hour6
final.at[p, 'Hour 7'] = hour7
final.at[p, 'Hour 8'] = hour8
final.at[p, 'Hour 9'] = hour9


# In[ ]:


p=138

hour1= ( final.iloc[p-18,2] + final.iloc[p+18,2] )/2
hour2= ( final.iloc[p-18,3] + final.iloc[p+18,3] )/2
hour3= ( final.iloc[p-18,4] + final.iloc[p+18,4] )/2
hour4= ( final.iloc[p-18,5] + final.iloc[p+18,5] )/2
hour5= ( final.iloc[p-18,6] + final.iloc[p+18,6] )/2
hour6= ( final.iloc[p-18,7] + final.iloc[p+18,7] )/2
hour7= ( final.iloc[p-18,8] + final.iloc[p+18,8] )/2
hour8= ( final.iloc[p-18,9] + final.iloc[p+18,9] )/2
hour9= ( final.iloc[p-18,10] + final.iloc[p+18,10] )/2



final.at[p, 'Hour 1'] = hour1
final.at[p, 'Hour 2'] = hour2
final.at[p, 'Hour 3'] = hour3
final.at[p, 'Hour 4'] = hour4
final.at[p, 'Hour 5'] = hour5
final.at[p, 'Hour 6'] = hour6
final.at[p, 'Hour 7'] = hour7
final.at[p, 'Hour 8'] = hour8
final.at[p, 'Hour 9'] = hour9


# In[ ]:


p=2352

hour1= ( final.iloc[p-18,2] + final.iloc[p+18,2] )/2
hour2= ( final.iloc[p-18,3] + final.iloc[p+18,3] )/2
hour3= ( final.iloc[p-18,4] + final.iloc[p+18,4] )/2
hour4= ( final.iloc[p-18,5] + final.iloc[p+18,5] )/2
hour5= ( final.iloc[p-18,6] + final.iloc[p+18,6] )/2
hour6= ( final.iloc[p-18,7] + final.iloc[p+18,7] )/2
hour7= ( final.iloc[p-18,8] + final.iloc[p+18,8] )/2
hour8= ( final.iloc[p-18,9] + final.iloc[p+18,9] )/2
hour9= ( final.iloc[p-18,10] + final.iloc[p+18,10] )/2



final.at[p, 'Hour 1'] = hour1
final.at[p, 'Hour 2'] = hour2
final.at[p, 'Hour 3'] = hour3
final.at[p, 'Hour 4'] = hour4
final.at[p, 'Hour 5'] = hour5
final.at[p, 'Hour 6'] = hour6
final.at[p, 'Hour 7'] = hour7
final.at[p, 'Hour 8'] = hour8
final.at[p, 'Hour 9'] = hour9


# In[ ]:


final.iloc[p,:]


# In[ ]:


rows_with_missing_values = final[final.isnull().any(axis=1)]
print(rows_with_missing_values) #All NA Values have been filled


# In[ ]:


# calculate the means for each subset of columns
final["Mean1"] = final.iloc[:, 2:4].mean(axis=1)
final["Mean2"] = final.iloc[:, 5:7].mean(axis=1)
final["Mean3"] = final.iloc[:, 8:10].mean(axis=1)


# In[ ]:


# Feature selection: Aggregation (Method 1- Take 2) Take mean of the 3hours data as a feature for each variable to predict 10 hours)
final.columns = [col.strip() for col in final.columns]
final['ItemName']=final['ItemName'].str.strip()
# Create a list of column names
columns = ['AMB_TEMPa', 'CH4a', 'COa', 'NMHCa', 'NOa', 'NO2a', 'NOXa', 'O3a', 'PM10a', 'PM2.5a','RAINFALLa', 'RHa', 'SO2a', 'THCa', 'WD_HRa', 'WIND_DIRECa', 'WIND_SPEEDa','WS_HRa','AMB_TEMPb', 'CH4b', 'COb', 'NMHCb', 'NOb', 'NO2b', 'NOXb', 'O3b', 'PM10b', 'PM2.5b','RAINFALLb', 'RHb', 'SO2b', 'THCb', 'WD_HRb', 'WIND_DIRECb', 'WIND_SPEEDb','WS_HRb','AMB_TEMPc', 'CH4c', 'COc', 'NMHCc', 'NOc', 'NO2c', 'NOXc', 'O3c', 'PM10c', 'PM2.5c','RAINFALLc', 'RHc', 'SO2c', 'THCc', 'WD_HRc', 'WIND_DIRECc', 'WIND_SPEEDc','WS_HRc']
# Create an empty DataFrame with the specified columns
final1 = pd.DataFrame(columns=columns)


# insert Id column at beginning
final1.insert(0, 'id', range(0, 244))


# iterate through each id
for i in range(244):
    id_df = final.loc[final['Index'] == i] # select rows with the iterated id value
    for index, row in id_df.iterrows():
        item_name = row['ItemName']
        mean1 =row['Mean1']
        mean2 =row['Mean2']
        mean3=row['Mean3']
        final1.loc[final1['id'] == i,item_name +('a')] = mean1 
        final1.loc[final1['id'] == i,item_name+('b')] = mean2
        final1.loc[final1['id'] == i,item_name+('c')] = mean3     

        
# print the new dataset
print(final1)


# In[ ]:


X = final1.drop(['id'],axis=1)
X = np.array(X)
# Normalize the features to have zero mean and unit variance
X = (X - np.mean(X, axis=0)) / np.std(X.astype(float), axis=0)


# In[ ]:


#Initialize y predicted values
predicted= np.zeros(X.shape[0])


for j in range(X.shape[0]):
    y_pred = np.dot(w, X[j]) + b
    predicted[j]=y_pred
 


# Print the final bias and weight values
print('Final bias:', b)
print('Final weights:', w)


# In[ ]:


import csv
# Transpose the array
predicted = np.transpose(predicted)



# Open the CSV file in write mode
with open('outputfinal7.csv', mode='w', newline='') as file:

    # Create a CSV writer object
    writer = csv.writer(file)

    # Write the array as a row
    writer.writerow(predicted)


# In[572]:


#NO REGULARIZATION

#Reduced data 
reduced_data = method1.drop(['id','Output'],axis=1)
reduced_data = np.array(reduced_data)

#Performing Linear Regression
y  = method1['Output']
y = np.array(y)
X= reduced_data

pred = np.zeros(X.shape[0])
# Normalize the features to have zero mean and unit variance
X = (X - np.mean(X, axis=0)) / np.std(X.astype(float), axis=0)
# Define initial bias, weight, learning rate, iteration, and regularization parameter
b = 0.0
w = np.zeros(X.shape[1])
lr = 0.03
iteration = 8000000


# Initialize Adagrad learning rate parameters
lr_b = 0.0
lr_w = np.zeros(X.shape[1])

# Perform stochastic gradient descent with Adagrad
for i in range(iteration):
    # Randomly select a single sample
    j = np.random.randint(X.shape[0])
    
    y_pred = np.dot(w, X[j]) + b
    b_grad = -2.0*(y[j] - y_pred) * 1
    w_grad = -2.0*(y[j] - y_pred) * X[j]
        
    
    # Update Adagrad learning rate parameters
    lr_b += b_grad ** 2
    lr_w = lr_w+ w_grad ** 2
    
    # Update bias and weight using Adagrad learning rate
    b -= lr / np.sqrt(lr_b) * b_grad
    w = w- lr / ((lr_w)**0.5) * w_grad

# Print the final bias and weight values
print('Final bias:', b)
print('Final weights:', w)

# Calculate the predicted values using the final bias and weights
pred = np.dot(X, w) + b
print(pred)



# In[ ]:


#RMSE: Above uses all training data to train the model
deviation=0
for j in range(X.shape[0]):
        y_pred = np.dot(w, X[j]) + b      
        deviation+= (y[j]-y_pred)**2
print(X.shape[0])
print(deviation)
RMSE= np.sqrt(deviation/X.shape[0])
print(RMSE)

