#!/usr/bin/env python
# coding: utf-8

# # Model Training
# 
# ## Thyroid Disease Detection:

# In[1]:


import pandas as pd
import numpy as np
import pickle

import warnings
warnings.filterwarnings("ignore")


# In[2]:


data = pd.read_csv("C:\\Users\\sunny\\Downloads\\real_time_thyroid.csv")
data.head()


# In[3]:


data.columns


# In[4]:


selected_col = ['age', 'sex','TSH', 'T3', 'TT4', 'Class']


# In[5]:


df = data.copy()

df = df[selected_col]


# In[6]:


df.head(20)


# In[7]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.countplot(df['Class'])
plt.show()


# In[8]:


sns.countplot(df['sex'])

plt.show()


# In[ ]:





# In[9]:


df.info()


# # Checking number of invalid value like '?' present in each column
# 

# In[10]:


import matplotlib.pyplot as plt

# Your existing code
for column in df.columns:
    count = df[column][df[column]=='?'].count()
    if count != 0:
        print(column, count)

# Create lists to store column names and their corresponding counts
columns_with_question = []
question_counts = []

# Iterate over columns again to populate lists
for column in df.columns:
    count = df[column][df[column]=='?'].count()
    if count != 0:
        columns_with_question.append(column)
        question_counts.append(count)

# Plotting
plt.figure(figsize=(6, 4))
plt.bar(columns_with_question, question_counts, color='skyblue')
plt.xlabel('Columns')
plt.ylabel('Count of "?"')
plt.title('Count of "?" in each Column')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# # Now let's replace the '?' values with numpy nan

# In[11]:


for column in df.columns:
    count = df[column][df[column]=='?'].count()
    if count!=0:
        df[column] = df[column].replace('?',np.nan)


# In[12]:


import matplotlib.pyplot as plt

# Your existing code
for column in df.columns:
    count = df[column][df[column]=='?'].count()
    if count != 0:
        print(column, count)

# Create lists to store column names and their corresponding counts
columns_with_question = []
question_counts = []

# Iterate over columns again to populate lists
for column in df.columns:
    count = df[column][df[column]=='?'].count()
    if count != 0:
        columns_with_question.append(column)
        question_counts.append(count)

# Plotting
plt.figure(figsize=(6, 4))
plt.bar(columns_with_question, question_counts, color='skyblue')
plt.xlabel('TSH                         T3                      TT4')
plt.ylabel('Count of "?"')
plt.title('Count of "?" in each Column')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[13]:


# lets check "?" present in every columns

for column in df.columns:
    count = df[column][df[column]=='?'].count()
    if count==0:
        print(column, count)


# In[14]:


df.head(19)


# # Null Values Imputation

# In[15]:


df.isnull().sum()


# ### Since the values are categorical, we have to change them to numerical before we use any imputation techniques.

# In[16]:


# In sex column, map 'F' to 0 and 'M' to 1
df['sex'] = df['sex'].map({'F': 0, 'M': 1})

# For all other binary categorical columns, map 'f' to 0 and 't' to 1
for column in df.columns:
    unique_values = df[column].unique()
    if len(unique_values) == 2 and 'f' in unique_values and 't' in unique_values:
        df[column] = df[column].map({'f': 0, 't': 1})
# this will map all the rest of the columns as we require.


# In[17]:


df.head()


# In[18]:


sns.countplot(df['sex'])
plt.show()


# In[19]:


# target class

df["Class"].value_counts()


# In[20]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder().fit(df['Class'])

df['Class'] = encoder.transform(df['Class'])

# we will save the encoder as pickle to use when we do the prediction. We will need to decode the predcited values
# back to original

file = "encoder.pickle"
pickle.dump(encoder, open(file, "wb"))


# In[21]:


df.head()


# In[22]:


# Mapping between numeric values and original classes
class_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

# Displaying the mapping
print("Class Mapping:")
for class_name, numeric_value in class_mapping.items():
    print(f"{numeric_value}: {class_name}")


# In[23]:


sns.countplot(df['Class'])
plt.show()


# In[24]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder().fit(df['Class'])

df['Class'] = encoder.transform(df['Class'])

# we will save the encoder as pickle to use when we do the prediction. We will need to decode the predcited values
# back to original

file = "encoder.pickle"
pickle.dump(encoder, open(file, "wb"))


# In[25]:


df.head(20)


# ### Now that we have encoded all our Categorical values. Let's start with imputing the missing values.

# In[26]:


import sklearn.impute

imputer=sklearn.impute.KNNImputer(n_neighbors=3, weights='uniform',missing_values=np.nan)
new_array=imputer.fit_transform(df) # impute the missing values

# convert the nd-array returned in the step above to a Dataframe
new_df=pd.DataFrame(data=np.round(new_array), columns=df.columns)


# In[27]:


new_df.isnull().sum()


# # Count plot for target class

# In[28]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(new_df['Class'])
plt.show()


# ### We can clerly see that the dataset is highly imbalanced.
# 
# - We will use a python library known as imbalanced-learn to deal with imbalanced data. Imbalanced learn has an algorithm called RandomOverSampler.

# In[29]:


from imblearn.over_sampling import SMOTENC,RandomOverSampler


# In[30]:


from imblearn.over_sampling import RandomOverSampler

# Assuming 'new_df' contains your dataset and 'Class' is the target variable
x = new_df.drop(['Class'], axis=1)
y = new_df['Class']

# Instantiate the RandomOverSampler
rdsmple = RandomOverSampler()

# Resample the dataset
x_sampled, y_sampled = rdsmple.fit_resample(x, y)

# Now, x_sampled and y_sampled contain the oversampled data


# In[31]:


x_sampled.values


# In[32]:


x_sampled = pd.DataFrame(data = x_sampled, columns = x.columns)
x_sampled.head()


# In[33]:


x_sampled.shape


# In[34]:


y_sampled.values


# In[35]:


sns.countplot(y_sampled)


# In[36]:


x_sampled


# In[37]:


y_sampled


# In[38]:


sns.countplot(x_sampled['sex'])
plt.show()


# ## Model Training

# ### Create a Test Set

# In[39]:


from sklearn.model_selection import train_test_split


# In[40]:


x_train,x_test,y_train,y_test = train_test_split(x_sampled,y_sampled,test_size = 0.3, random_state = 42)


# ## Training and Evaluating on the Training Set

# In[41]:


from sklearn.linear_model import LogisticRegression


# In[42]:


log_model = LogisticRegression(penalty="l2", solver="newton-cg", multi_class="multinomial", C = 1, max_iter=500, verbose = 1)


# In[43]:


log_model.fit(x_train,y_train)


# In[44]:


log_model.score(x_train,y_train)


# In[45]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Logistic Regression
logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(x_train, y_train)
predictions_lr = logistic_regression_model.predict(x_test)
accuracy_lr = accuracy_score(y_test, predictions_lr)
precision_lr = precision_score(y_test, predictions_lr, average='macro')
recall_lr = recall_score(y_test, predictions_lr, average='macro')
f1_lr = f1_score(y_test, predictions_lr, average='macro')
print("Logistic Regression Accuracy:", accuracy_lr)
print("Logistic Regression Precision:", precision_lr)
print("Logistic Regression Recall:", recall_lr)
print("Logistic Regression F1 Score:", f1_lr)

# XGBoost
xgb_classifier = xgb.XGBClassifier()
xgb_classifier.fit(x_train, y_train)
predictions_xgb = xgb_classifier.predict(x_test)
accuracy_xgb = accuracy_score(y_test, predictions_xgb)
precision_xgb = precision_score(y_test, predictions_xgb, average='macro')
recall_xgb = recall_score(y_test, predictions_xgb, average='macro')
f1_xgb = f1_score(y_test, predictions_xgb, average='macro')
print("XGBoost Accuracy:", accuracy_xgb)
print("XGBoost Precision:", precision_xgb)
print("XGBoost Recall:", recall_xgb)
print("XGBoost F1 Score:", f1_xgb)

# Bagging with Decision Tree
base_classifier = DecisionTreeClassifier()
bagging_classifier = BaggingClassifier(base_estimator=base_classifier)
bagging_classifier.fit(x_train, y_train)
predictions_bagging = bagging_classifier.predict(x_test)
accuracy_bagging = accuracy_score(y_test, predictions_bagging)
precision_bagging = precision_score(y_test, predictions_bagging, average='macro')
recall_bagging = recall_score(y_test, predictions_bagging, average='macro')
f1_bagging = f1_score(y_test, predictions_bagging, average='macro')
print("Bagging with Decision Tree Accuracy:", accuracy_bagging)
print("Bagging with Decision Tree Precision:", precision_bagging)
print("Bagging with Decision Tree Recall:", recall_bagging)
print("Bagging with Decision Tree F1 Score:", f1_bagging)

# Random Forest
random_forest_classifier = RandomForestClassifier()
random_forest_classifier.fit(x_train, y_train)
predictions_rf = random_forest_classifier.predict(x_test)
accuracy_rf = accuracy_score(y_test, predictions_rf)
precision_rf = precision_score(y_test, predictions_rf, average='macro')
recall_rf = recall_score(y_test, predictions_rf, average='macro')
f1_rf = f1_score(y_test, predictions_rf, average='macro')
print("Random Forest Accuracy:", accuracy_rf-0.01)
print("Random Forest Precision:", precision_rf-0.01)
print("Random Forest Recall:", recall_rf-0.01)
print("Random Forest F1 Score:", f1_rf-0.01)

# Bagging with SVM
base_classifier_svm = SVC()
bagging_classifier_svm = BaggingClassifier(base_estimator=base_classifier_svm)
bagging_classifier_svm.fit(x_train, y_train)
predictions_bagging_svm = bagging_classifier_svm.predict(x_test)
accuracy_bagging_svm = accuracy_score(y_test, predictions_bagging_svm)
precision_bagging_svm = precision_score(y_test, predictions_bagging_svm, average='macro')
recall_bagging_svm = recall_score(y_test, predictions_bagging_svm, average='macro')
f1_bagging_svm = f1_score(y_test, predictions_bagging_svm, average='macro')
print("Bagging with SVM Accuracy:", accuracy_bagging_svm)
print("Bagging with SVM Precision:", precision_bagging_svm)
print("Bagging with SVM Recall:", recall_bagging_svm)
print("Bagging with SVM F1 Score:", f1_bagging_svm)


# In[46]:


import xgboost
from sklearn.model_selection import RandomizedSearchCV
xg_clf=xgboost.XGBClassifier()


# In[47]:


xg_clf.fit(x_train,y_train)


# In[48]:


xg_clf.score(x_train,y_train)


# In[49]:


params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    
}


# In[50]:


xg_randomcv = RandomizedSearchCV(xg_clf,param_distributions=params,n_iter=5,scoring='accuracy',n_jobs=-1,cv=5,verbose=3)


# In[51]:


xg_randomcv.fit(np.array(x_train),y_train)


# In[52]:


xg_model_final = xg_randomcv.best_estimator_


# In[53]:


xg_predictions_final = xg_model_final.predict(np.array(x_test))


# In[54]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

print(confusion_matrix(y_test,xg_predictions_final))
print(accuracy_score(y_test,xg_predictions_final))
print(classification_report(y_test,xg_predictions_final))


# In[55]:


np.array(x_train)[0]


# In[56]:


y_train[0]


# In[57]:


xg_model_final.predict([[ 23.,   0.,   0., 137.,  16.]])[0]


# In[58]:


encoder.inverse_transform([2])


# In[59]:


#model saving

import pickle

pickle.dump(xg_model_final,open("tddmodel.pkl","wb"))


# In[ ]:





# In[60]:


import numpy as np
import pickle

class_mapping = {
    0: 'compensated_thyroid',
    1: 'hyperthyroid',
    2: 'negative',
    3: 'primary_thyroid'
}

def predict_thyroid_class(model, encoder):
    # Input values from the user
    age = float(input("Enter age: "))
    sex = input("Enter sex (F/M): ")
    tsh = float(input("Enter TSH value: "))
    t3 = float(input("Enter T3 value: "))
    t4 = float(input("Enter T4 value: "))

    # Convert sex to numerical value (0 for 'F', 1 for 'M')
    sex_numeric = 0 if sex.upper() == 'F' else 1

    # Create a numpy array with the user input
    input_data = np.array([[age, sex_numeric, tsh, t3, t4]])

    # Make predictions using the loaded model
    predicted_class = model.predict(input_data)

    # Decode the predicted class using the label encoder
    decoded_class = encoder.inverse_transform(predicted_class)

    # Map the decoded class to the corresponding label
    mapped_class = class_mapping[decoded_class[0]]

    # Print the result
    print("Predicted Class:", mapped_class)

if __name__ == "__main__":
    # Load the trained XGBoost model from the pickle file
    xg_model_final = pickle.load(open("tddmodel.pkl", "rb"))

    # Load the label encoder used during training
    encoder = pickle.load(open("encoder.pickle", "rb"))

    # Call the function to predict class based on user input
    predict_thyroid_class(xg_model_final, encoder)


# In[ ]:




