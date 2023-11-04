#!/usr/bin/env python
# coding: utf-8

# <div style='font-size:30px;line-height:40px;text-align:center'><strong>Assignment 2</strong></div>

# <div style='font-size:40px;line-height:40px;text-align:center'> Predicting Amphibian Survival Based on Environmental Features: Using Semi-Supervised Learning </div>

# # Dataset

# >The dataset is derived from GIS, satellite, and environmental impact assessment reports for two road projects in Poland. It includes attributes related to water reservoirs and their surroundings, and target labels representing seven amphibian species. The goal is to predict amphibian presence using GIS and satellite features, contributing to understanding their habitat preferences and aiding conservation efforts in areas affected by road development. The research paper titled "Predicting presence of amphibian species using features obtained from GIS and satellite images" by Marcin Blachnik, Marek Sołtysiak, and Dominika Dąbrowska provides insights into this study.

# # Specification of dataset

# >The dataset contains 189 instances and 22 attributes, including 18 quantitative and 4 qualitative attributes. 
#  
# >The attributes include information about the geographical location of the water reservoir, the altitude, the type of water reservoir, and various environmental factors such as temperature, precipitation, and humidity.
# 
# >The dataset also includes information about the presence of 7 different amphibian species near the water reservoirs.
# 
# >The dataset was created by combining two datasets: one containing information about the water reservoirs and the other containing information about the amphibian species
# 
# >The dataset has no missing values.

# # Preprocessing

# >Preprocessing is a critical step in the data analysis and machine learning workflow. It involves transforming raw data into a clean, structured, and informative format that can be used for analysis and model training. The main goals of data preprocessing are to handle missing or inconsistent data, eliminate noise, and prepare the data for further analysis or modeling. 

# In[1]:


#Basic imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import warnings


# # Data Extraction

# In[2]:


df=pd.read_csv('C:\\Users\\ASUS\\Downloads\\amphibians\\datasetNew.csv',sep=';')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# # Data validation and cleansing

# ## Check for null values

# In[6]:


df.isnull().sum()


# ## Checking for duplicates

# In[7]:


df.duplicated().sum()


# ## Checking datatypes

# In[8]:


df.dtypes


# In[9]:


df['Motorway']


# ## Renaming columns

# In[10]:


# Creating a dictionary to map old column names to new column names
column_mapping = {
    'SR': 'Surface of water',
    'NR': 'Number of reservoirs',
    'TR': 'Type of reservoirs',
    'SUR1':'Surrounding land 1',
    'SUR2':'Surrounding land 2',
    'SUR3':'Surrounding land 3',
    'UR':'Use of reservoir',
    'VR':'Vegetation',
    'FR':'Fishing',
    'OR':'Water percent access',
    'RR':'Road Distance',
    'BR':'Building Distance',
    'MR':'Reservoir Status',
    'CR':'Shore Type',
    'Label 1':'Green Frogs',
    'Label 2':'Brown frogs',
    'Label 3':'Common toad',
    'Label 4':'Fire-bellied toad',
    'Label 5':'Tree frog',
    'Label 6':'Common newt',
    'label 7':'Great crested newt',
}



# In[11]:


# Use the 'rename()' method to rename the columns
df.rename(columns=column_mapping, inplace=True)


# In[12]:


df.tail()


# ## Identifying Outliers

# In[13]:


# Select numerical columns for outlier detection
numerical_columns = ['Surface of water', 'Number of reservoirs', 'Water percent access', 'Road Distance']

# Create box plots for each numerical column
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[numerical_columns])
plt.title("Box Plot of Numerical Columns")
plt.xlabel("Numerical Columns")
plt.ylabel("Values")
plt.show()


# ### Calculating unique values in numerical columns

# In[14]:


df['Number of reservoirs'].value_counts()


# In[15]:


df['Water percent access'].value_counts()


# In[16]:


df['Road Distance'].value_counts()


# In[17]:


df['Surface of water'].value_counts()


# ### Handling outliers where necessary

# In[18]:


# Calculate the IQR for each numerical column
Q1 = df[numerical_columns].quantile(0.25)
Q3 = df[numerical_columns].quantile(0.75)
IQR = Q3 - Q1

# Calculate the lower and upper bounds for outliers
lower_bound = Q1 - 10 * IQR
upper_bound = Q3 + 10 * IQR

# Identify outliers in each numerical column
outliers = ((df[numerical_columns] < lower_bound) | (df[numerical_columns] > upper_bound)).any(axis=1)

# Optionally, you can print or further investigate the outliers
df[outliers].head()





# ## Plotting the covariance matrix

# In[19]:


corr = df.corr()
corr.style.background_gradient(cmap='coolwarm')


# # Data Aggregation and Representation

# # Problem 1 : Finding how many types of amphibians are present?

# ## Feature Engineering

# ### Creating a new column named amphibians

# In[20]:


frogs_sum= df['Green frogs']+df['Brown frogs']+df['Tree frog']
toads_sum=df['Common toad']+df['Fire-bellied toad']
newts_sum=df['Common newt']+df['Great crested newt']

# Replace values greater than or equal to 1 with 1
frogs_sum = frogs_sum.apply(lambda x: 1 if x >= 1 else x)
toads_sum = toads_sum.apply(lambda x: 1 if x >= 1 else x)
newts_sum = newts_sum.apply(lambda x: 1 if x >= 1 else x)

amphibians_sum=frogs_sum + toads_sum + newts_sum
df['amphibians']=amphibians_sum


# In[21]:


df.head()


# ## Shuffling the dataset for even distribution of labels
# 

# In[22]:


df = shuffle(df, random_state=42)


# ## Creating x and y sets

# In[23]:


X=df.drop(['ID','Motorway','amphibians','Green frogs','Brown frogs','Common toad','Fire-bellied toad','Tree frog','Common newt','Great crested newt','amphibians'],axis=1)
y=df['amphibians']


# ## Splitting the dataset

# In[24]:


# Set a random seed for numpy (to ensure reproducibility)
np.random.seed(42)

#  Split Data into Training and Test Sets (20% for the test set)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Split Training Data into Labeled and Unlabeled Sets (80% for unlabeled data)
X_train_labeled, X_train_unlabeled, y_train_labeled, y_train_unlabeled = train_test_split(
    X_train, y_train, test_size=0.8, random_state=42)


# # How classification and semi supervised algorithmns work 

# Classification algorithms are a type of supervised learning algorithm used for predictive modeling tasks, where the goal is to assign a class label to input data based on the patterns observed in the training data. Here's how classification algorithms work:
# 
# >Data Preparation: The labeled dataset is divided into training and testing sets. The training set is used to train the classifier, while the testing set is used to evaluate its performance.
# 
# >Model Training: The classifier learns from the training data by extracting patterns and relationships between input features and their corresponding class labels. This process involves adjusting the model's parameters to minimize the prediction errors.
# 
# >Model Evaluation: The trained classifier is tested on the testing set to measure its accuracy and performance. Evaluation metrics such as accuracy, precision, recall, F1-score, and ROC curves are used to assess the model's effectiveness.

# Semi-supervised algorithms, on the other hand, deal with partially labeled data, where only a subset of the data has known class labels. The key idea behind semi-supervised learning is to utilize both the labeled and unlabeled data to improve model performance. Here's how semi-supervised algorithms work:
# 
# >Data Preparation: Similar to supervised learning, the labeled dataset is divided into training and testing sets. However, in semi-supervised learning, a portion of the data remains unlabeled.
# 
# >Label Propagation: Semi-supervised algorithms use the labeled data to propagate labels to the unlabeled data based on similarity or clustering techniques. The assumption is that similar data points should have similar labels.
# 
# >Model Training: The classifier is trained on the combined labeled and pseudo-labeled data (data with propagated labels). The unlabeled data contributes to a better representation of the underlying data distribution, potentially improving the model's performance.
# 
# >Model Evaluation: The performance of the semi-supervised classifier is evaluated on the testing set, similar to supervised learning, using various evaluation metrics.

# # Data Analysis

# ## Model Application : GradientBoostingClassifier

# In[25]:


from sklearn.semi_supervised import LabelSpreading
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from tabulate import tabulate
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# In[26]:


# Initialize and Train the Label Spreading Model with a higher number of neighbors
label_spread_model = LabelSpreading(kernel='knn', n_neighbors=30, max_iter=100)
label_spread_model.fit(X_train_labeled, y_train_labeled)

# Propagate Labels to Unlabeled Data
y_propagated = label_spread_model.predict(X_train_unlabeled)

# Combine Labeled and Propagated Data
X_train_combined = pd.concat([X_train_labeled, X_train_unlabeled])
y_train_combined = pd.concat([y_train_labeled, pd.Series(y_propagated)])

# Train Semi-Supervised Model on Combined Data
semi_supervised_classifier = GradientBoostingClassifier(random_state=42)
semi_supervised_classifier.fit(X_train_combined, y_train_combined)

#  Make Predictions on Test Set
y_pred = semi_supervised_classifier.predict(X_test)

# Evaluate the Semi-Supervised Model's Performance on Test Set
accuracy = accuracy_score(y_test, y_pred)
print("Test Set Accuracy:", accuracy)


# ### Tabulating the results

# In[27]:


# Tabulate y_test and y_pred
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(tabulate(results, headers='keys', tablefmt='pretty', showindex=False))


# ### Plotting the confusion matrix

# In[28]:


# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
# Create a figure and axes
fig, ax = plt.subplots(figsize=(8, 6))

# Create ConfusionMatrixDisplay object and plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=semi_supervised_classifier.classes_)
disp.plot(cmap='Blues', ax=ax)


# # Problem 2 : Checking if any type of amphibians are present or not?

# ### Modifying the amphibians column

# In[29]:


# Replacing all values greater thn or equal to 1 with 1
df['amphibians']=amphibians_sum.apply(lambda x: 1 if x >= 1 else x)


# In[30]:


df['amphibians'].value_counts()


# ### Applying the model again

# In[31]:


df['amphibians']=amphibians_sum.apply(lambda x: 1 if x >= 1 else x)

X=df.drop(['ID','Motorway','amphibians','Green frogs','Brown frogs','Common toad','Fire-bellied toad','Tree frog','Common newt','Great crested newt','amphibians'],axis=1)
y=df['amphibians']

# Set a random seed for numpy (to ensure reproducibility)
np.random.seed(42)

# Split Data into Training and Test Sets (20% for the test set)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Split Training Data into Labeled and Unlabeled Sets (50% for unlabeled data)
X_train_labeled, X_train_unlabeled, y_train_labeled, y_train_unlabeled = train_test_split(
    X_train, y_train, test_size=0.5, random_state=42)

# Initialize and Train the Label Spreading Model with a higher number of neighbors
label_spread_model = LabelSpreading(kernel='knn', n_neighbors=50, max_iter=100)
label_spread_model.fit(X_train_labeled, y_train_labeled)

# Propagate Labels to Unlabeled Data
y_propagated = label_spread_model.predict(X_train_unlabeled)

# Combine Labeled and Propagated Data
X_train_combined = pd.concat([X_train_labeled, X_train_unlabeled])
y_train_combined = pd.concat([y_train_labeled, pd.Series(y_propagated)])

# Train Semi-Supervised Model on Combined Data
semi_supervised_classifier = GradientBoostingClassifier(random_state=42)
semi_supervised_classifier.fit(X_train_combined, y_train_combined)

# Make Predictions on Test Set
y_pred = semi_supervised_classifier.predict(X_test)

# Evaluate the Semi-Supervised Model's Performance on Test Set
accuracy = accuracy_score(y_test, y_pred)
print("Test Set Accuracy:", accuracy)


# ### Plotting the confusion matrix

# In[32]:


# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
# Create a figure and axes
fig, ax = plt.subplots(figsize=(8, 6))

# Create ConfusionMatrixDisplay object and plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=semi_supervised_classifier.classes_)
disp.plot(cmap='Blues', ax=ax)


# In[33]:


import matplotlib.pyplot as plt


# Get unique labels in the "amphibians" column (assuming you have only 2 amphibian labels)
labels = df['amphibians'].unique()

# Count the occurrences of each label in y_pred and y_test
pred_counts = df['amphibians'].value_counts().reindex(labels, fill_value=0)
test_counts = y_test.value_counts().reindex(labels, fill_value=0)

# Set the x-axis positions for the bars
x = range(len(labels))

# Create the bar plots for y_pred and y_test
plt.bar(x, pred_counts, width=0.4, align='center', label='y_pred')
plt.bar([pos + 0.4 for pos in x], test_counts, width=0.4, align='center', label='y_test')

# Set the x-axis labels and plot title
plt.xticks(x, labels)
plt.xlabel('Amphibian Labels')
plt.ylabel('Count')
plt.title('Comparison of y_pred and y_test')

# Add legend and show the plot
plt.legend()
plt.show()


# # Problem 3: Finding out the categories of amphibians in each reservoir

# ### Modifying the target column: Creating new column

# In[34]:


df.head()


# In[35]:


# Creating amphibians_new column containing 6 different categories 
frogs_sum_new= df['Green frogs']+df['Brown frogs']+df['Tree frog']
toads_sum_new=df['Common toad']+df['Fire-bellied toad']
newts_sum_new=df['Common newt']+df['Great crested newt']

# Replace values greater than or equal to 1 with 1
frogs_sum_new = frogs_sum_new.apply(lambda x: 'frog ' if x >= 1 else "")
toads_sum_new = toads_sum_new.apply(lambda x: 'toad ' if x >= 1 else '')
newts_sum_new = newts_sum_new.apply(lambda x: 'newt ' if x >= 1 else '')

amphibians_sum_new=frogs_sum_new + toads_sum_new + newts_sum_new
df['amphibians_new']=amphibians_sum_new


# In[36]:


df.head()


# In[37]:


df['amphibians_new'].value_counts()


# ### Encoding ambhibians_new column

# In[38]:


data = {
    "amphibians_new": ["frog toad", "frog toad newt", "frog", "", "toad", "frog newt"],
    "counts": [74, 57, 43, 6, 5, 4]
}

# Create a Label Encoding dictionary for the unique values in the column, including empty instances
label_encoding_dict = {}
label = 0
for value in df['amphibians_new'].unique():
    if value not in label_encoding_dict:
        label_encoding_dict[value] = label
        label += 1

# Apply the Label Encoding to the column
df['encoded_amphibians'] = df['amphibians_new'].map(label_encoding_dict)

df.head()


# In[39]:


df['encoded_amphibians'].value_counts()


# ### Again applying model

# In[40]:


X=df.drop(['ID','Motorway','amphibians','Green frogs','Brown frogs','Common toad','Fire-bellied toad','Tree frog','Common newt','Great crested newt','amphibians','amphibians_new','encoded_amphibians'],axis=1)
y=df['encoded_amphibians']

# Set a random seed for numpy (to ensure reproducibility)
np.random.seed(42)

# Split Data into Training and Test Sets (20% for the test set)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Split Training Data into Labeled and Unlabeled Sets (50% for unlabeled data)
X_train_labeled, X_train_unlabeled, y_train_labeled, y_train_unlabeled = train_test_split(
    X_train, y_train, test_size=0.8, random_state=42)

# Initialize and Train the Label Spreading Model with a higher number of neighbors
label_spread_model = LabelSpreading(kernel='knn', n_neighbors=10, max_iter=100)
label_spread_model.fit(X_train_labeled, y_train_labeled)

# Propagate Labels to Unlabeled Data
y_propagated = label_spread_model.predict(X_train_unlabeled)

# Combine Labeled and Propagated Data
X_train_combined = pd.concat([X_train_labeled, X_train_unlabeled])
y_train_combined = pd.concat([y_train_labeled, pd.Series(y_propagated)])

# Train Semi-Supervised Model on Combined Data
semi_supervised_classifier = GradientBoostingClassifier(random_state=42)
semi_supervised_classifier.fit(X_train_combined, y_train_combined)

# Make Predictions on Test Set
y_pred = semi_supervised_classifier.predict(X_test)

# Evaluate the Semi-Supervised Model's Performance on Test Set
accuracy = accuracy_score(y_test, y_pred)
print("Test Set Accuracy:", accuracy)


# # Problem 4: Predicting 3 target columns namely 'frogs','toads','newts'

# ### Creating 3 new binary features 'frogs','toads','newts'

# In[41]:


frogs_sum1= df['Green frogs']+df['Brown frogs']+df['Tree frog']
toads_sum1=df['Common toad']+df['Fire-bellied toad']
newts_sum1=df['Common newt']+df['Great crested newt']

# Replace values greater than or equal to 1 with 1
df['frogs'] = frogs_sum1.apply(lambda x: 1 if x >= 1 else x)
df['toads'] = toads_sum1.apply(lambda x: 1 if x >= 1 else x)
df['newts'] = newts_sum1.apply(lambda x: 1 if x >= 1 else x)
df.head()


# In[60]:


df['frogs'].value_counts()


# In[61]:


df['toads'].value_counts()


# In[62]:


df['newts'].value_counts()


# ## Approach 1: Applying Label Propogation to predict pseudo labels

# In[42]:


from sklearn.cluster import KMeans


# In[43]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.semi_supervised import LabelPropagation
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier


# Remove any irrelevant columns, including 'ID', 'Motorway', etc.
X = df.drop(['ID', 'Motorway', 'amphibians', 'Green frogs', 'Brown frogs',
             'Common toad', 'Fire-bellied toad', 'Tree frog', 'Common newt',
             'Great crested newt', 'amphibians', 'amphibians_new',
             'encoded_amphibians', 'frogs', 'toads', 'newts'], axis=1)
y = df[['frogs', 'toads', 'newts']]

# Set a random seed for reproducibility
np.random.seed(42)

# Split Data into Training and Test Sets (20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Split Training Data into Labeled and Unlabeled Sets (80% for unlabeled data)
X_train_labeled, X_train_unlabeled, y_train_labeled, y_train_unlabeled = train_test_split(
    X_train, y_train, test_size=0.8, random_state=42)

# Initial Label Propagation on Labeled Data
label_prop_model = LabelPropagation()
label_prop_model.fit(X_train_labeled, y_train_labeled['frogs'])  # Fit LabelPropagation for 'frogs' target

# Pseudo-labeling and Confidence thresholding using Label Propagation
# Predict pseudo-labels for the unlabeled data using the trained Label Propagation model
pseudo_labels_frogs = label_prop_model.predict(X_train_unlabeled)

# Compute the confidence of the pseudo-labels (proportion of data points for each class)
confidence = np.max(label_prop_model.predict_proba(X_train_unlabeled), axis=1)

# Apply confidence threshold of 0.8 to select confident pseudo-labels
threshold = 0.8
confident_mask = confidence >= threshold
print(confident_mask,'confident_mask')
# Filter confident pseudo-labels and their corresponding features
X_unlabeled_confident = X_train_unlabeled[confident_mask]
y_unlabeled_confident = y_train_unlabeled[confident_mask]

# Use Label Propagation to predict pseudo-labels for unlabeled confident data
label_prop_model_confident = LabelPropagation()
label_prop_model_confident.fit(X_train_labeled, y_train_labeled['frogs'])  # Fit LabelPropagation for 'frogs' target
pseudo_labels_frogs_confident = label_prop_model_confident.predict(X_unlabeled_confident)

# Convert pseudo-labels from Label Propagation to binary format for each target variable
pseudo_labels_binary = []
for i in range(y.shape[1]):
    pseudo_labels = np.zeros(len(pseudo_labels_frogs_confident), dtype=int)
    pseudo_labels[pseudo_labels_frogs_confident == i] = 1
    pseudo_labels_binary.append(pseudo_labels)

# Combine labeled and pseudo-labeled data for the final model
X_combined_final = np.vstack([X_train_labeled, X_unlabeled_confident])
y_combined_final = np.vstack([y_train_labeled.to_numpy(), np.array(pseudo_labels_binary).T])

# Train the final K-nearest neighbors classifier using the combined dataset from all iterations
final_knn_classifier = MultiOutputClassifier(KNeighborsClassifier(n_neighbors=5))
final_knn_classifier.fit(X_combined_final, y_combined_final)

# Evaluate the final KNN model on the test dataset (X_test)
y_test_predictions = final_knn_classifier.predict(X_test)

# Calculate accuracy for each target variable
accuracies = []
for i in range(y_test.shape[1]):
    accuracy = accuracy_score(y_test.iloc[:, i], y_test_predictions[:, i])
    accuracies.append(accuracy)
    print(f'Accuracy for {y_test.columns[i]}: {accuracy:.4f}')

# Calculate overall accuracy (average of target accuracies)
overall_accuracy = np.mean(accuracies)
print(f'Overall Accuracy: {overall_accuracy:.4f}')


# ### Calculating the hamming_loss and accuracy

# In[44]:


from sklearn.metrics import hamming_loss, accuracy_score

# Calculate the Hamming loss
hamming_loss_value = hamming_loss(y_test, y_test_predictions)
accuracy = 1 - hamming_loss_value

print("Hamming Loss:", hamming_loss_value)
print("Accuracy:", accuracy)


# In[45]:


from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, average_precision_score, matthews_corrcoef, cohen_kappa_score, mean_absolute_error, mean_squared_error


# ### Confusion matrix

# In[46]:


# Plot confusion matrix for each target variable
def plot_confusion_matrix(y_true, y_pred, target_names, title='Confusion Matrix'):
    for i in range(len(target_names)):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title + ' - ' + target_names[i])
        plt.colorbar()
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()

# Assuming you have the names of the target variables (frogs, toads, newts)
target_names = ['frogs', 'toads', 'newts']

# Convert y_test_predictions and y_test to NumPy arrays
y_test_predictions_array = np.array(y_test_predictions)
y_test_array = y_test.to_numpy()

# Plot confusion matrix
plot_confusion_matrix(y_test_array, y_test_predictions_array, target_names)


# ### Roc Curve

# In[47]:


from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score


# Convert the probabilities to binary predictions (assuming a threshold of 0.5 for each target variable)
y_test_predictions_binary = (y_test_predictions >= 0.5).astype(int)

# Calculate the confusion matrix for each target variable
target_names = ['frogs', 'toads', 'newts']
for i, target_name in enumerate(target_names):
    cm = confusion_matrix(y_test[target_name], y_test_predictions_binary[:, i])
    print(f'Confusion Matrix for {target_name}:')
    print(cm)

# Calculate and plot ROC curves for each target variable
plt.figure(figsize=(8, 6))
for i, target_name in enumerate(target_names):
    fpr, tpr, _ = roc_curve(y_test[target_name], y_test_predictions[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{target_name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc='lower right')
plt.show()

# Calculate the overall AUC score (average over all target variables)
overall_auc = roc_auc_score(y_test, y_test_predictions)
print(f'Overall AUC: {overall_auc:.2f}')


# ### Calculating various metrics

# In[48]:


conf_matrix = confusion_matrix(y_test.values.ravel(), y_test_predictions.ravel())

precision = precision_score(y_test.values.ravel(), y_test_predictions.ravel(), average='macro')
recall = recall_score(y_test.values.ravel(), y_test_predictions.ravel(), average='macro')
f1 = f1_score(y_test.values.ravel(), y_test_predictions.ravel(), average='macro')

roc_auc = roc_auc_score(y_test, y_test_predictions)
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1:.4f}')
print(f'AUC-ROC: {roc_auc:.4f}')


#  ## Approach :1 Hypertuning the model using RandomizedSearchCV

# In[49]:


from sklearn.model_selection import  RandomizedSearchCV


warnings.filterwarnings('ignore')

# Perform hyperparameter tuning for K-nearest neighbors using RandomizedSearchCV
param_distributions = {'estimator__n_neighbors': np.arange(1, 300)}
knn_classifier = KNeighborsClassifier()
final_knn_classifier = MultiOutputClassifier(knn_classifier)
random_search = RandomizedSearchCV(final_knn_classifier, param_distributions, n_iter=50, cv=5, scoring='accuracy', random_state=42)
random_search.fit(X_train_labeled, y_train_labeled)

# Print the best hyperparameter value found by RandomizedSearchCV
print("Best n_neighbors:", random_search.best_params_['estimator__n_neighbors'])

# Train the final K-nearest neighbors classifier using the combined dataset from all iterations
final_knn_classifier = MultiOutputClassifier(KNeighborsClassifier(n_neighbors=random_search.best_params_['estimator__n_neighbors']))
final_knn_classifier.fit(X_combined_final, y_combined_final)

# Evaluate the final KNN model on the test dataset (X_test)
y_test_predictions = final_knn_classifier.predict(X_test)

# Calculate accuracy for each target variable
accuracies = []
for i in range(y_test.shape[1]):
    accuracy = accuracy_score(y_test.iloc[:, i], y_test_predictions[:, i])
    accuracies.append(accuracy)
    print(f'Accuracy for {y_test.columns[i]}: {accuracy:.4f}')

# Calculate overall accuracy (average of target accuracies)
overall_accuracy = np.mean(accuracies)
print(f'Overall Accuracy: {overall_accuracy:.4f}')


# In[50]:


# Plot confusion matrix for each target variable
def plot_confusion_matrix(y_true, y_pred, target_names, title='Confusion Matrix'):
    for i in range(len(target_names)):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title + ' - ' + target_names[i])
        plt.colorbar()
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()

# Assuming you have the names of the target variables (frogs, toads, newts)
target_names = ['frogs', 'toads', 'newts']

# Convert y_test_predictions and y_test to NumPy arrays
y_test_predictions_array = np.array(y_test_predictions)
y_test_array = y_test.to_numpy()

# Plot confusion matrix
plot_confusion_matrix(y_test_array, y_test_predictions_array, target_names)


# ### Calculating various metrics

# In[51]:


conf_matrix = confusion_matrix(y_test.values.ravel(), y_test_predictions.ravel())

precision = precision_score(y_test.values.ravel(), y_test_predictions.ravel(), average='macro')
recall = recall_score(y_test.values.ravel(), y_test_predictions.ravel(), average='macro')
f1 = f1_score(y_test.values.ravel(), y_test_predictions.ravel(), average='macro')

roc_auc = roc_auc_score(y_test, y_test_predictions)
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1:.4f}')
print(f'AUC-ROC: {roc_auc:.4f}')


# ## Approach 2: Applying KMeans to predict pseudo labels

# In[52]:


from sklearn.cluster import KMeans


# Set a random seed for reproducibility
np.random.seed(42)

# Split Data into Training and Test Sets (20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Split data into 80% unlabeled and 20% labeled
X_train_labeled, X_train_unlabeled, y_train_labeled, y_train_unlabeled = train_test_split(
    X_train, y_train, test_size=0.8, random_state=42)

# Use K-Means clustering to predict pseudo-labels for unlabeled data
n_clusters = len(y.columns)  # Number of clusters = Number of target variables
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
pseudo_labels_kmeans = kmeans.fit_predict(X_train_unlabeled)

# Convert pseudo-labels from K-Means clustering to binary format for each target variable
pseudo_labels_binary = []
for i in range(n_clusters):
    pseudo_labels = np.zeros(len(y_train_unlabeled), dtype=int)
    pseudo_labels[pseudo_labels_kmeans == i] = 1
    pseudo_labels_binary.append(pseudo_labels)

# Combine labeled and pseudo-labeled data for the final model
X_combined_final = np.vstack([X_train_labeled, X_train_unlabeled])
y_combined_final = np.vstack([y_train_labeled.to_numpy(), np.array(pseudo_labels_binary).T])

# Train the final K-nearest neighbors classifier using the combined dataset from all iterations
final_knn_classifier = MultiOutputClassifier(KNeighborsClassifier(n_neighbors=5))
final_knn_classifier.fit(X_combined_final, y_combined_final)

# Evaluate the final KNN model on the test dataset (X_test)
y_test_predictions = final_knn_classifier.predict(X_test)

# Calculate accuracy for each target variable
accuracies = []
for i in range(y_test.shape[1]):
    accuracy = accuracy_score(y_test.iloc[:, i], y_test_predictions[:, i])
    accuracies.append(accuracy)
    print(f'Accuracy for {y_test.columns[i]}: {accuracy:.4f}')

# Calculate overall accuracy (average of target accuracies)
overall_accuracy = np.mean(accuracies)
print(f'Overall Accuracy: {overall_accuracy:.4f}')


# ### Confusion Matrix

# In[53]:


# Plot confusion matrix for each target variable
def plot_confusion_matrix(y_true, y_pred, target_names, title='Confusion Matrix'):
    for i in range(len(target_names)):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title + ' - ' + target_names[i])
        plt.colorbar()
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()

# Assuming you have the names of the target variables (frogs, toads, newts)
target_names = ['frogs', 'toads', 'newts']

# Convert y_test_predictions and y_test to NumPy arrays
y_test_predictions_array = np.array(y_test_predictions)
y_test_array = y_test.to_numpy()

# Plot confusion matrix
plot_confusion_matrix(y_test_array, y_test_predictions_array, target_names)


# ### Roc Curve

# In[54]:


# Convert the probabilities to binary predictions (assuming a threshold of 0.5 for each target variable)
y_test_predictions_binary = (y_test_predictions >= 0.5).astype(int)

# Calculate the confusion matrix for each target variable
target_names = ['frogs', 'toads', 'newts']
for i, target_name in enumerate(target_names):
    cm = confusion_matrix(y_test[target_name], y_test_predictions_binary[:, i])
    print(f'Confusion Matrix for {target_name}:')
    print(cm)

# Calculate and plot ROC curves for each target variable
plt.figure(figsize=(8, 6))
for i, target_name in enumerate(target_names):
    fpr, tpr, _ = roc_curve(y_test[target_name], y_test_predictions[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{target_name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc='lower right')
plt.show()

# Calculate the overall AUC score (average over all target variables)
overall_auc = roc_auc_score(y_test, y_test_predictions)
print(f'Overall AUC: {overall_auc:.2f}')


# ### Calculating various metrics

# In[55]:


conf_matrix = confusion_matrix(y_test.values.ravel(), y_test_predictions.ravel())

precision = precision_score(y_test.values.ravel(), y_test_predictions.ravel(), average='macro')
recall = recall_score(y_test.values.ravel(), y_test_predictions.ravel(), average='macro')
f1 = f1_score(y_test.values.ravel(), y_test_predictions.ravel(), average='macro')

roc_auc = roc_auc_score(y_test, y_test_predictions)
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1:.4f}')
print(f'AUC-ROC: {roc_auc:.4f}')


# ## Approach 2 : Hypertuning the KNN Classifier using RandomizedSearchCV

# In[56]:


# ignore all the warnings
warnings.filterwarnings('ignore')


# Use K-Means clustering to predict pseudo-labels for unlabeled data
n_clusters = len(y.columns)  # Number of clusters = Number of target variables
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
pseudo_labels_kmeans = kmeans.fit_predict(X_train_unlabeled)

# Convert pseudo-labels from K-Means clustering to binary format for each target variable
pseudo_labels_binary = []
for i in range(n_clusters):
    pseudo_labels = np.zeros(len(y_train_unlabeled), dtype=int)
    pseudo_labels[pseudo_labels_kmeans == i] = 1
    pseudo_labels_binary.append(pseudo_labels)

# Combine labeled and pseudo-labeled data for the final model
X_combined_final = np.vstack([X_train_labeled, X_train_unlabeled])
y_combined_final = np.vstack([y_train_labeled.to_numpy(), np.array(pseudo_labels_binary).T])

# Hyperparameter tuning for KNN Classifier using RandomizedSearchCV
knn_classifier = KNeighborsClassifier()
param_dist = {'n_neighbors': np.arange(1, 300)} 
random_search = RandomizedSearchCV(knn_classifier, param_distributions=param_dist, n_iter=5, cv=3)
random_search.fit(X_combined_final, y_combined_final)

# Get the best KNN model from the randomized search
best_knn_classifier = random_search.best_estimator_

# Evaluate the best KNN model on the test dataset (X_test)
y_test_predictions = best_knn_classifier.predict(X_test)

# Calculate accuracy for each target variable
accuracies = []
for i in range(y_test.shape[1]):
    accuracy = accuracy_score(y_test.iloc[:, i], y_test_predictions[:, i])
    accuracies.append(accuracy)
    print(f'Accuracy for {y_test.columns[i]}: {accuracy:.4f}')

# Calculate overall accuracy (average of target accuracies)
overall_accuracy = np.mean(accuracies)
print(f'Overall Accuracy: {overall_accuracy:.4f}')


# In[57]:


# Plot confusion matrix for each target variable
def plot_confusion_matrix(y_true, y_pred, target_names, title='Confusion Matrix'):
    for i in range(len(target_names)):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title + ' - ' + target_names[i])
        plt.colorbar()
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()

# Assuming you have the names of the target variables (frogs, toads, newts)
target_names = ['frogs', 'toads', 'newts']

# Convert y_test_predictions and y_test to NumPy arrays
y_test_predictions_array = np.array(y_test_predictions)
y_test_array = y_test.to_numpy()

# Plot confusion matrix
plot_confusion_matrix(y_test_array, y_test_predictions_array, target_names)


# ### Calculating various metrics

# In[58]:


conf_matrix = confusion_matrix(y_test.values.ravel(), y_test_predictions.ravel())

precision = precision_score(y_test.values.ravel(), y_test_predictions.ravel(), average='macro')
recall = recall_score(y_test.values.ravel(), y_test_predictions.ravel(), average='macro')
f1 = f1_score(y_test.values.ravel(), y_test_predictions.ravel(), average='macro')

roc_auc = roc_auc_score(y_test, y_test_predictions)
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1:.4f}')
print(f'AUC-ROC: {roc_auc:.4f}')


# # Result of Implementation

# In[59]:


from tabulate import tabulate
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

# Sample data for demonstration purposes (replace with actual data)
model1_metrics = {
"Confusion Matrix":[[26 ,11],[24 ,53]],
"Precision": 0.6741,
"Recall": 0.6955,
"F1-Score": 0.6747,
"AUC-ROC": 0.6171,
}

model2_metrics = {
"Confusion Matrix":[[26 ,11],[24, 53]],
"Precision": 0.6741,
"Recall": 0.6955,
"F1-Score": 0.6747,
"AUC-ROC": 0.6171,

}


model3_metrics = {
"Confusion Matrix":[[35 , 2],[38 ,39]],
"Precision": 0.7153,
"Recall": 0.7262,
"F1-Score": 0.6487,
"AUC-ROC": 0.5073,
}



model4_metrics = {
"Confusion Matrix":[[36 , 1],[40 ,37]],
"Precision": 0.7237,
"Recall": 0.7267,
"F1-Score": 0.6403,
"AUC-ROC": 0.5000,

}


# Combine the metrics for both models into a list of dictionaries
metrics_list = [
    {"Model": "Label Propogation", **model1_metrics},
    {"Model": "Label Propogation with hypertuning", **model2_metrics},
    {"Model": "Kmeans", **model3_metrics}, 
    {"Model": "Kmeans with hypertuning", **model4_metrics},
]

# Convert the list of dictionaries into a table
table = tabulate(metrics_list, headers="keys", tablefmt="pretty")

# Print the table
print(table)


# # Conclusion

# >Label Propagation Model:
# The Label Propagation model achieved moderate performance in predicting the target variables 'frogs,' 'toads,' and 'newts.' It had a precision of 0.6741, a recall of 0.6955, and an F1-Score of 0.6747. The AUC-ROC score was 0.6171.
# 
# >Label Propagation Model with Hyperparameter Tuning:
# Hyperparameter tuning did not significantly improve the performance of the Label Propagation model. The model achieved a precision of 0.6741, a recall of 0.6955, and an F1-Score of 0.6747. The AUC-ROC score remained at 0.6171.
# 
# >Kmeans Model:
# The Kmeans model also showed moderate performance in predicting the target variables. It had a precision of 0.7153, a recall of 0.7262, and an F1-Score of 0.6487. The AUC-ROC score was 0.5073.
# 
# >Kmeans Model with Hyperparameter Tuning:
# Similar to the Label Propagation model, hyperparameter tuning did not significantly improve the Kmeans model's performance. The model achieved a precision of 0.7237, a recall of 0.7267, and an F1-Score of 0.6403. The AUC-ROC score was 0.5000
# 
# Overall, the models' performance might not be entirely satisfactory, and there is room for improvement. Further exploration of other algorithms, feature engineering, or using different clustering or semi-supervised techniques could be considered to enhance the models' accuracy and predictive capabilities. Additionally, evaluating the models on a larger and more diverse dataset may provide better insights into their effectiveness.
# 

# In[ ]:




