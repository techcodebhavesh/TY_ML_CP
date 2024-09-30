# %% [markdown]
# #Disease Prediction from Symptoms

# %% [markdown]
# The dataset source: http://people.dbmi.columbia.edu/~friedma/Projects/DiseaseSymptomKB/index.html
# 
# The extraction was performed by copying the data on the website in the .html format and saving it in an Excel file for performing further operations. Basic cleaning, segmentation of columns and string formatting were performed in excel. The excel sheet was then added to this Google Colab Notebook.
# 
# 

# %% [markdown]
# ##Importing all needed libraries

# %%
# Import Dependencies
import csv
import pandas as pd
import numpy as np
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# %% [markdown]
# ## Transforming & Loading the Data

# %% [markdown]
# ### Loading the Dataset File generated after preprocessing in excel

# %%
disease_list = []

def return_list(disease):
    disease_list = []
    match = disease.replace('^','_').split('_') # using _ as common splitting delimeter
    ctr = 1
    for group in match:
        if ctr%2==0:
            disease_list.append(group) # refer the data format
        ctr = ctr + 1

    return disease_list

with open("D:/TY_ML_CP/Disease-Prediction-from-Symptoms-main/Datasets/raw_data_2.csv") as csvfile:
    reader = csv.reader(csvfile)
    disease=""
    weight = 0
    disease_list = []
    dict_wt = {}
    dict_=defaultdict(list)
    
    for row in reader:

        if row[0]!="\xc2\0xa0" and row[0]!="": # for handling file encoding errors
          # saving disease and frequency
            disease = row[0]
            disease_list = return_list(disease)
            weight = row[1]

        if row[2]!="\xc2\0xa0" and row[2]!="":
            symptom_list = return_list(row[2])

            for d in disease_list:
                for s in symptom_list:
                    dict_[d].append(s) # adding all symptoms
                dict_wt[d] = weight


# %% [markdown]
# ### Reformatting the data

# %%
# saving cleaned data
with open("dataset_clean.csv","w") as csvfile:
    writer = csv.writer(csvfile)
    for key,values in dict_.items():
        for v in values:
            #key = str.encode(key)
            key = str.encode(key).decode('utf-8')
            #.strip()
            #v = v.encode('utf-8').strip()
            #v = str.encode(v)
            writer.writerow([key,v,dict_wt[key]])

# %%
columns = ['Source','Target','Weight'] # source: disease, target: symptom, weight: number of cases

# %%
data = pd.read_csv("dataset_clean.csv",names=columns, encoding ="ISO-8859-1")

# %%
data.head()

# %%
data.to_csv("dataset_clean.csv",index=False)

# %%
data = pd.read_csv("dataset_clean.csv", encoding ="ISO-8859-1")

# %%
data.head()

# %%
len(data['Source'].unique()) # unique diseases

# %%
len(data['Target'].unique()) # unique symptoms

# %%
df = pd.DataFrame(data)

# %%
df_1 = pd.get_dummies(df.Target) # 1 hot encoding symptoms

# %%
df_1.head()

# %%
df.head()

# %%
df_s = df['Source']

# %%
df_pivoted = pd.concat([df_s,df_1], axis=1)

# %%
df_pivoted.drop_duplicates(keep='first',inplace=True)

# %%
df_pivoted[:5]

# %%
len(df_pivoted)

# %%
cols = df_pivoted.columns

# %%
cols = cols[1:] # removing headings

# %%
# visualizing existance of symptoms for diseases
df_pivoted = df_pivoted.groupby('Source').sum()
df_pivoted = df_pivoted.reset_index()
df_pivoted[:5]

# %%
len(df_pivoted)

# %%
df_pivoted.to_csv("df_pivoted.csv")

# %%
# defining data for training
x = df_pivoted[cols]
y = df_pivoted['Source']

# %% [markdown]
# ##Building Model

# %%
# importing all needed libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# %%
# performing train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# %%
# Training multinomial naive bayes
mnb = MultinomialNB()
mnb = mnb.fit(x_train, y_train)

# %%
mnb.score(x_test, y_test)

# %% [markdown]
# ###Inferences on train and test split
# It can't work on unseen data because it has never seen that disease before. Also, there is only one point for each disease and hence no point for this. So we need to train the model entirely. Then what will we test it on? Missing data? Say given one symptom what is the disease? This is again multilabel classification. We can work symptom on symptom. What exactly is differential diagnosis, we need to replicate that.

# %%
mnb_tot = MultinomialNB()
mnb_tot = mnb_tot.fit(x, y)

# %%
mnb_tot.score(x, y)

# %%
disease_pred = mnb_tot.predict(x)

# %%
disease_real = y.values

# %%
# printing model error
for i in range(0, len(disease_real)):
    if disease_pred[i]!=disease_real[i]:
        print ('Pred: {0} Actual:{1}'.format(disease_pred[i], disease_real[i]))

# %% [markdown]
# ## Using decision trees

# %%
from sklearn.tree import DecisionTreeClassifier, export_graphviz

# %%
print ("DecisionTree")
dt = DecisionTreeClassifier()
clf_dt=dt.fit(x,y)
print ("Acurracy: ", clf_dt.score(x,y))

# %%
from sklearn import tree 
from sklearn.tree import export_graphviz

export_graphviz(dt, 
                out_file='tree.jpg', 
                feature_names=cols
               )

# %%
from IPython.display import Image
Image(filename='tree.jpg')

# %% [markdown]
# ##Manual Analysis data

# %%
data = pd.read_csv("D:/TY_ML_CP/Disease-Prediction-from-Symptoms-main/Datasets/Training.csv")

# %%
data.head()

# %%
data.columns

# %%
len(data.columns)

# %%
len(data['prognosis'].unique())

# %%
df = pd.DataFrame(data)

# %%
df.head()

# %%
len(df)

# %%
cols = df.columns

# %%
cols = cols[:-1]

# %%
len(cols)

# %%
x = df[cols]
y = df['prognosis']

# %%
x

# %%
y

# %% [markdown]
# ## Building Classifier: Using Multinomial Naive Bayes

# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# %%
mnb = MultinomialNB()
mnb = mnb.fit(x_train, y_train)

# %%
mnb.score(x_test, y_test)

# %%
from sklearn import model_selection
print ("cross result========")
scores = model_selection.cross_val_score(mnb, x_test, y_test, cv=3)
print (scores)
print (scores.mean())

# %%
test_data = pd.read_csv("D:/TY_ML_CP/Disease-Prediction-from-Symptoms-main/Datasets/Testing.csv")

# %%
test_data.head()

# %%
testx = test_data[cols]
testy = test_data['prognosis']

# %%
mnb.score(testx, testy)

# %%
from sklearn import model_selection
print ("cross result========")
scores = model_selection.cross_val_score(mnb, x_test, y_test, cv=3)
print (scores)
print (scores.mean())

# %% [markdown]
# ##Building Classifier: Using Decision Trees

# %%
from sklearn.tree import DecisionTreeClassifier, export_graphviz

# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# %%
print ("DecisionTree")
dt = DecisionTreeClassifier(min_samples_split=20)
clf_dt=dt.fit(x_train,y_train)
print ("Acurracy: ", clf_dt.score(x_test,y_test))

# %%
from sklearn import model_selection
print ("cross result========")
scores = model_selection.cross_val_score(dt, x_test, y_test, cv=3)
print (scores)
print (scores.mean())

# %%
print ("Acurracy on the actual test data: ", clf_dt.score(testx,testy))

# %%
from sklearn import tree 
from sklearn.tree import export_graphviz

export_graphviz(dt, 
                out_file='tree.dot', 
                feature_names=cols)

# %%
!dot -Tpng tree.dot -o tree.png

# %%
from IPython.display import Image
Image(filename='tree.png')

# %%


# %%
dt.__getstate__()

# %%


# %% [markdown]
# ##Finding Feature Importances

# %%
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

importances = dt.feature_importances_
print(dt.feature_importances_)

# %%
indices = np.argsort(importances)[::-1]
print([data.columns[i] for i in indices])

# %%
features = cols

# %%
for f in range(20):
    print("%d. feature %d - %s (%f)" % (f + 1, indices[f], features[indices[f]] ,importances[indices[f]]))

# %%
export_graphviz(dt, 
                out_file='tree-top5.dot', 
                feature_names=cols,
                max_depth = 5
               )

# %%
!dot -Tpng tree-top5.dot -o tree-top5.png

# %%
from IPython.display import Image
Image(filename='tree-top5.png')

# %%
feature_dict = {}
for i,f in enumerate(features):
    feature_dict[f] = i

# %%
feature_dict['internal_itching']

# %%
sample_x = [i/52 if i==52 else i*0 for i in range(len(features))]
cols = list(data.columns) 
print(cols.index('skin_rash'))

# %% [markdown]
# This means predicting the disease where the only symptom is redness_of_eyes.

# %%
sample_x = np.array(sample_x).reshape(1,len(sample_x))

# %%
dt.predict(sample_x)

# %%
dt.predict_proba(sample_x)

# %%
len(sample_x)

# %%
symptoms = ['skin_rash','itching','nodal_skin_eruptions','increased_appetite','irritability']
ipt = [0 for i in range(len(features))]
for s in symptoms:
  ipt[cols.index(s)]=1
ipt = np.array([ipt])
print(ipt)
print(dt.predict(ipt))
dt.predict_proba(ipt)

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
df = pd.read_csv('D:/TY_ML_CP/Disease-Prediction-from-Symptoms-main/Datasets/Training.csv')

#drop  column named prognosis
df = df.drop('prognosis', axis=1)


#print column names
cols = df.columns
print(cols)
# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[cols])

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42)  # You can adjust n_clusters based on your dataset
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualize the clusters using PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Cluster'], cmap='viridis')
plt.title('Patient Clusters Based on Symptoms')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')
plt.show()


# %%
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("D:/TY_ML_CP/Disease-Prediction-from-Symptoms-main/Datasets/Training.csv")

# Drop the label column if included (like 'prognosis')
X = data.drop('prognosis', axis=1)

# Apply K-means clustering
kmeans = KMeans(n_clusters=5, random_state=42)  # choose number of clusters (k)
kmeans.fit(X)

# Add the cluster labels to the data
data['Cluster'] = kmeans.labels_

# Visualize the clustering (2D projection using PCA)
from sklearn.decomposition import PCA

pca = PCA(2)
X_pca = pca.fit_transform(X)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=data['Cluster'], cmap='rainbow')
plt.title('Patient Clusters')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()


# %%
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Scale the features for DBSCAN
X_scaled = StandardScaler().fit_transform(X)

# Fit DBSCAN (adjust eps and min_samples accordingly)
dbscan = DBSCAN(eps=0.5, min_samples=5)
data['Cluster_DBSCAN'] = dbscan.fit_predict(X_scaled)

print(data['Cluster_DBSCAN'].value_counts())


# %%
from sklearn.decomposition import PCA

# Apply PCA to reduce dimensions to 2
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Now use X_pca for visualization instead of X
plt.figure(figsize=(10, 7))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=data['Cluster'], cmap='viridis', alpha=0.6)
plt.title('K-Means Clustering with PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Load the dataset
data = pd.read_csv("D:/TY_ML_CP/Disease-Prediction-from-Symptoms-main/Datasets/Training.csv")

# Drop the label column if included (like 'prognosis')
X = data.drop('prognosis', axis=1)

# Feature Scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(X)

# Determine the optimal number of clusters using the Elbow method
inertia = []
silhouette_scores = []
for k in range(2, 15):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(scaled_data, kmeans.labels_))

# Plot the Elbow method and Silhouette score
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(2, 15), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')

plt.subplot(1, 2, 2)
plt.plot(range(2, 15), silhouette_scores, marker='o')
plt.title('Silhouette Scores')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')

plt.show()

# Fit the KMeans model with the optimal number of clusters
optimal_k = np.argmax(silhouette_scores) + 2  # Adjust for range starting at 2
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_data)

# Save the clustered data to CSV
data.to_csv("D:/TY_ML_CP/Disease-Prediction-from-Symptoms-main/Datasets/Clustered_data.csv", index=False)

# Visualize clusters (2D projection using PCA)
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_data)

plt.figure(figsize=(10, 7))
scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=data['Cluster'], cmap='viridis')
plt.title('Patient Clusters')
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.colorbar(scatter, label='Cluster')
plt.show()


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

# Load the dataset
data = pd.read_csv("D:/TY_ML_CP/Disease-Prediction-from-Symptoms-main/Datasets/Training.csv")

# Drop the label column if included (like 'prognosis')
X = data.drop('prognosis', axis=1)

# Feature Scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(X)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
pca_result = pca.fit_transform(scaled_data)

# Fit DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)  # You can adjust eps and min_samples as needed
clusters = dbscan.fit_predict(pca_result)

# Add the cluster labels to the original data
data['Cluster'] = clusters

# Save the clustered data to a new CSV file
data.to_csv("D:/TY_ML_CP/Disease-Prediction-from-Symptoms-main/Datasets/DBSCAN_Clustered_data_with_PCA.csv", index=False)

# Visualize the clusters (2D projection)
plt.figure(figsize=(10, 7))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=clusters, cmap='viridis', marker='o', edgecolor='k', s=50)
plt.title('DBSCAN Clustering with PCA Reduction')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster Label')
plt.show()


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("D:/TY_ML_CP/Disease-Prediction-from-Symptoms-main/Datasets/Training.csv")

# Display the first few rows of the dataset
print(data.head())

# Summary statistics
print(data.describe())

# Check for missing values
print(data.isnull().sum())

# Visualize the distribution of each feature using histograms
plt.figure(figsize=(15, 10))
data.hist(bins=30, figsize=(15, 10), grid=False)
plt.suptitle('Feature Distribution', fontsize=20)
plt.tight_layout()
plt.show()

# Visualize the correlations between features using a heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Heatmap', fontsize=20)
plt.show()

# Scatter plot of the first two features
plt.figure(figsize=(10, 7))
plt.scatter(data.iloc[:, 0], data.iloc[:, 1], alpha=0.6)
plt.title('Scatter Plot of Feature 1 vs Feature 2', fontsize=20)
plt.xlabel('Feature 1', fontsize=14)
plt.ylabel('Feature 2', fontsize=14)
plt.grid()
plt.show()

# Box plot to visualize the spread of features
plt.figure(figsize=(15, 8))
sns.boxplot(data=data.drop('prognosis', axis=1))  # Exclude the label column
plt.title('Box Plot of Features', fontsize=20)
plt.xticks(rotation=45)
plt.show()

# Pairplot to visualize relationships between features
sns.pairplot(data, hue='prognosis', diag_kind='kde', markers='o')
plt.title('Pairplot of Features Colored by Prognosis', fontsize=20)
plt.show()



