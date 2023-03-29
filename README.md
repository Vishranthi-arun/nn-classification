# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
![image](https://user-images.githubusercontent.com/93427278/228422828-2d732363-c4ce-445e-9944-fc9d06db5846.png)
<br>

## DESIGN STEPS
### STEP 1:
Import the necessary packages & modules
### STEP 2:
Load and read the dataset
### STEP 3:
Perform pre processing and clean the dataset
### STEP 4:
Encode categorical value into numerical values using ordinal/label/one hot encoding
### STEP 5:
 Visualize the data using different plots in seaborn
### STEP 6:
 Normalize the values and split the values for x and y
### STEP 7:
 Build the deep learning model with appropriate layers and depth
### STEP 8:
 Analyze the model using different metrics
### STEP 9:
Plot a graph for Training Loss, Validation Loss Vs Iteration & for Accuracy, Validation Accuracy vs Iteration
### STEP 10:
Save the model using pickle
### STEP 11:
Using the DL model predict for some random inputs

## PROGRAM
```
Developed by : Vishranthi A
Reg no. 212221230124
```
```python
import pandas as pd
df=pd.read_csv('/content/customers (1).csv')
df.head()
df.columns
df.dtypes
df.shape
clean=df.dropna(axis=0)
clean.isnull().sum()
clean.shape
clean.dtypes
clean['Gender'].unique()
clean['Ever_Married'].unique()
clean['Graduated'].unique()
clean['Profession'].unique()
clean['Spending_Score'].unique()
clean['Var_1'].unique()
clean['Segmentation'].unique()
from sklearn.preprocessing import OrdinalEncoder
categorylist=[['Male', 'Female'],
              ['No', 'Yes'],
              ['No', 'Yes'],
              ['Healthcare', 'Engineer', 'Lawyer', 'Artist', 'Doctor',
       'Homemaker', 'Entertainment', 'Marketing', 'Executive'],
              ['Low', 'High', 'Average']]
enc=OrdinalEncoder(categories=categorylist)
cust1=clean.copy()
# INPUT---> OrdinalEncoder
cust1[['Gender',
        'Ever_Married',
        'Graduated',
        'Profession',
        'Spending_Score']]=enc.fit_transform(cust1[['Gender',
                                                    'Ever_Married',
                                                    'Graduated',
                                                    'Profession',
                                                    'Spending_Score']])
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
cust1.dtypes
#OUTPUT--> LabelEncoder
cust1['Segmentation'] = le.fit_transform(cust1['Segmentation'])
cust1=cust1.drop('ID',axis=1)
cust1=cust1.drop('Var_1',axis=1)
cust1.dtypes
import seaborn as sns
# Calculate the correlation matrix
corr = cust1.corr()

# Plot the heatmap
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        cmap="BuPu",
        annot= True)
import matplotlib.pylab as plt
sns.pairplot(cust1)
sns.displot(cust1['Age'])
plt.figure(figsize=(10,6))
sns.countplot(cust1['Family_Size'])
plt.figure(figsize=(10,6))
sns.boxplot(x='Family_Size',y='Age',data=cust1)
plt.figure(figsize=(10,6))
sns.scatterplot(x='Family_Size',y='Spending_Score',data=cust1)
X=cust1[['Gender','Ever_Married','Age','Graduated','Profession','Work_Experience','Spending_Score','Family_Size']].values
y1=cust1[['Segmentation']].values
from sklearn.preprocessing import OneHotEncoder
one_hot_enc = OneHotEncoder()
one_hot_enc.fit(y1)
y1.shape
y = one_hot_enc.transform(y1).toarray()
y.shape
y1[0]
y[0]
X.shape
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=50)
X_train[0]
X_train.shape
from sklearn.preprocessing import MinMaxScaler
scaler_age = MinMaxScaler()
scaler_age.fit(X_train[:,2].reshape(-1,1))
import numpy as np
X_train_scaled = np.copy(X_train)
X_test_scaled = np.copy(X_test)
# To scale the Age column
X_train_scaled[:,2] = scaler_age.transform(X_train[:,2].reshape(-1,1)).reshape(-1)
X_test_scaled[:,2] = scaler_age.transform(X_test[:,2].reshape(-1,1)).reshape(-1)
# Creating the model
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
ai_brain = Sequential([
  # Develop your model here
  Dense(8,input_shape=(8,)),
  Dense(12,activation='relu'),
  Dense(16,activation='relu'),
  Dense(4,activation='softmax')
])
ai_brain.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=2)
ai_brain.fit(x=X_train_scaled,y=y_train,
             epochs=2000,batch_size=256,
             validation_data=(X_test_scaled,y_test),
             )
metrics = pd.DataFrame(ai_brain.history.history)
metrics.head()
metrics[['loss','val_loss']].plot()
# Sequential predict_classes function is deprecated
# predictions = ai_brain.predict_classes(X_test)
x_test_predictions = np.argmax(ai_brain.predict(X_test_scaled), axis=1)
x_test_predictions.shape
y_test_truevalue = np.argmax(y_test,axis=1)
y_test_truevalue.shape
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test_truevalue,x_test_predictions))
print(classification_report(y_test_truevalue,x_test_predictions))
ai_brain.save('customer_classification_model.h5')
import pickle
with open('customer_data.pickle', 'wb') as fh:
   pickle.dump([X_train_scaled,y_train,X_test_scaled,y_test,cust1,clean,scaler_age,enc,one_hot_enc,le], fh)
from tensorflow.keras.models import load_model
ai_brain = load_model('customer_classification_model.h5')
# Loading the data
with open('customer_data.pickle', 'rb') as fh:
   [X_train_scaled,y_train,X_test_scaled,y_test,customers_1,customer_df_cleaned,scaler_age,enc,one_hot_enc,le]=pickle.load(fh)
# Prediction for a single input
x_single_prediction = np.argmax(ai_brain.predict(X_test_scaled[1:2,:]), axis=1)
print(x_single_prediction)
print(le.inverse_transform(x_single_prediction))
```
## Dataset Information
![dataset](https://user-images.githubusercontent.com/93427278/228417612-5ecdae3c-c62e-49b8-a479-f6faa9a567fc.png)
<br>

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
![image](https://user-images.githubusercontent.com/93427278/228422604-340573e9-fa81-4607-9811-7652f6775f39.png)

<br>

### Classification Report
![image](https://user-images.githubusercontent.com/93427278/228422560-64dc8003-3d92-44a7-ba32-5bcf85753885.png)
<br>

### Confusion Matrix
![image](https://user-images.githubusercontent.com/93427278/228422488-29d3d625-708c-4165-a732-906215013c64.png)
<br>


### New Sample Data Prediction
![image](https://user-images.githubusercontent.com/93427278/228422415-ea53ff1b-7513-4c9e-8543-8b8974f1d5bb.png)

## RESULT
Thus, a Simple Neural Network Classification Model is developed successfully.
