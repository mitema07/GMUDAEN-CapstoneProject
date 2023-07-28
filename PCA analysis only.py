#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import os
import re
import glob
import pandas as pd


# In[2]:


os.getcwd()


# In[3]:


os.chdir('C:\RotorCraftData\DAEN690-Karate_Choppers')
print(os.getcwd())


# In[5]:


# this is the path to the folder where you have the CSVs, NO OTHER CSVs SHOULD BE PRESENT
# please make sure this path is not inside the scope of GitHub so we do not go over on data for our repo
path = r'C:\RotorCraftData\CSV'
pattern = r'.*2023\.06\.15.*\.csv$'

# this imports a list of columns that was saved after the removal of variance on a single CSV, this list will be used to define which columns to read in
with open('src./use_cols_aws.pkl', 'rb') as f: 
    use_cols = pickle.load(f)

# the data will be labeled using the information from the flight logs
label_table = pd.DataFrame({
    'Date': ['2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15', '2023-06-15'], 
    # Replace with actual start time of maneuvers
    'StartTime': ['13:22:15.0', '13:25:25.0', '13:29:25.0', '11:56:25.0', '11:58:03.0', '11:59:51.0', '16:10:04.0', '16:11:41.0', '16:14:20.0', '13:43:12.0', '13:44:10.0', '13:45:19.0', '12:08:11.0', '12:09:31.0', '12:10:51.0', '16:34:28.0', '16:35:06.0', '16:38:26.0'],
    # Replace with actual end time of maneuvers
    'EndTime': ['13:22:25.0', '13:25:38.0', '13:29:40.0', '11:56:38.0', '11:58:24.0', '12:00:00.0', '16:10:12.0', '16:11:46.0', '16:14:29.0', '13:43:35.0', '13:44:18.0', '13:45:30.0', '12:08:35.0', '12:09:52.0', '12:11:18.0', '16:34:42.0', '16:35:27.0', '16:38:36.0'],
    'Label': ['Dynamic Rollover', 'Dynamic Rollover', 'Dynamic Rollover', 'Dynamic Rollover', 'Dynamic Rollover', 'Dynamic Rollover', 'Dynamic Rollover', 'Dynamic Rollover', 'Dynamic Rollover', 'LOW-G', 'LOW-G', 'LOW-G', 'LOW-G', 'LOW-G', 'LOW-G', 'LOW-G', 'LOW-G', 'LOW-G']  
})

# convert date, start time, and end time columns to datetime type
label_table['Date'] = pd.to_datetime(label_table['Date'])
label_table['StartTime'] = pd.to_datetime(
    label_table['StartTime'], format='%H:%M:%S.%f').dt.strftime('%H:%M:%S.%f')
label_table['EndTime'] = pd.to_datetime(
    label_table['EndTime'], format='%H:%M:%S.%f').dt.strftime('%H:%M:%S.%f')


def combine_csv_files(csv_directory, columns_to_use, label_df):
    # get list of CSV file paths in the directory
    csv_files = [os.path.join(csv_directory, filename) for filename in os.listdir(
        csv_directory) if re.match(pattern, filename)]
    # create an empty dataframe to store the combined data
    combined_df = pd.DataFrame()

    # iterate over each CSV file
    for file in csv_files:
        # read CSV file and select desired columns
        temp_df = pd.read_csv(file, usecols=columns_to_use, names=columns_to_use, skiprows=3, skipfooter=1, engine='python')
        # drop rows that Elapsed Time are mostly null, these are the breaks in simulation
        temp_df.dropna(subset=['Elapsed Time'], inplace=True)
        # temp_df.drop(['Elapsed Time'], inplace=True)
        temp_df.dropna(inplace=True)
        # concatenate the temporary dataframe with the running dataframe
        combined_df = pd.concat([combined_df, temp_df], ignore_index=True)

    # convert the time column on original df to correct format
    combined_df['System UTC Time'] = pd.to_datetime(
    combined_df['System UTC Time'], format='%H:%M:%S.%f').dt.strftime('%H:%M:%S.%f')
    # convert the date column on original df to correct format
    combined_df['Date'] = pd.to_datetime(combined_df['Date'])
    
    # apply the labeling to the dataset
    for _, row in label_df.iterrows():
        # extract date, start time, and end time from the current row
        date = row['Date']
        start_time = row['StartTime']
        end_time = row['EndTime']
        label = row['Label']

        # filter the existing dataset based on matching date and within start time and end time
        filter_condition = (combined_df['Date'] == date) & (
            combined_df['System UTC Time'].between(start_time, end_time))
        combined_df.loc[filter_condition, 'Label'] = label
    dummies_df = pd.get_dummies(combined_df['Label'], dummy_na=False)
    dummies_df = dummies_df.astype(int)
    combined_df = pd.concat([combined_df, dummies_df], axis=1)
    # Convert the time column to pandas datetime format if it's not already in that format
    combined_df['System UTC Time'] = pd.to_datetime(combined_df['System UTC Time'], format='%H:%M:%S.%f')

    # Set the start and end time range
    start_time = pd.to_datetime('11:56:25.0', format='%H:%M:%S.%f')
    end_time = pd.to_datetime('16:38:26.0', format='%H:%M:%S.%f')

    # Filter the DataFrame to include rows between the start and end times
    combined_df = combined_df[(combined_df['System UTC Time'] >= start_time) & (combined_df['System UTC Time'] <= end_time)].copy()

    combined_df.drop(['Elapsed Time', 'Date', 'System UTC Time', 'Label'], inplace=True, axis=1)
    
    return combined_df

# this calls the function from above that cleans and creates dummy variables for our target variables
df = combine_csv_files(path, use_cols, label_table)
# this will create a pickle file with the working dataframe in your directory with the original CSV files
# you will not need to run this script again, as we will load in the dataframe from the pickle file
with open(f'{path}/working_df_aws.pkl', 'wb') as f:
    pickle.dump(df, f)


# In[6]:


df.shape


# In[7]:


df.head()


# In[8]:


df.describe()


# In[9]:


df.shape


# In[10]:


df['GPS 1 NAV ID'].fillna("0", inplace = True)


# In[11]:


df['NAV 2 NAV ID'].fillna("0", inplace = True)


# In[12]:


numeric_columns = list(df.select_dtypes('float64','int64'))
variance = df[numeric_columns].var()
# Setting a threshold for variance
threshold = 0.01
# Cols with near zero variance
near_zero_variance_columns = variance[variance <= threshold].index.tolist()
print(near_zero_variance_columns)
len(near_zero_variance_columns)


# In[13]:


#Assigning df columns to cols variable.
cols = df.columns
cols


# In[14]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[15]:


#Code to plot the correlation matrix with heatmap. It is very similar to the Cov_matrix one, it is very crowded.
corrmat1 = df.corr()
top_corr_features = corrmat1.index
plt.figure(figsize=(30,30))
g=sns.heatmap(df[top_corr_features].corr(), annot=True, cmap="RdYlGn")


# In[ ]:


print(corrmat1)


# In[ ]:


#Based on the df corr matrix, we can see that there are two cols with zero varianec. I will remove them. 


# In[16]:


df2 = df.drop(['GPS 1 DME Time', 'NAV 2 DME Time'], axis = 1)


# In[17]:


df2.shape


# # PCA analysis

# In[18]:


#https://www.edureka.co/blog/principal-component-analysis/
# Starting the PCA with a dataset with 26 cols including the response variables. 
#Step 1 - import required packages
import sklearn
from sklearn.preprocessing import StandardScaler
from matplotlib import rcParams
from sklearn.decomposition import PCA
from sklearn import decomposition
from sklearn.preprocessing import scale
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.cm import register_cmap
from scipy import stats
from sklearn.decomposition import PCA as sklearnPCA
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[19]:


#Step 2- Select df. I will be using df2 which has 26 variables including the response variables. That dataset has
# only numeric values and the date which I could convert to float. System UTC time could not be
#converted to float. I need to research more. 

df2.head()


# In[20]:


print(df.dtypes.value_counts())


# In[21]:


#Step3 - formatting the data. It has been formatted already. Excluding response variables.  , 'label_0','label_Dynamic Rollover','label_LOW-G'.
# df2 has only float and int data columns. Dropping Label, Dynamic Rollover and LOW-G
df2 = df.drop(['Dynamic Rollover','LOW-G'], axis=1)


# In[22]:


# Step 4: Standardization. It run using df2
X_std = StandardScaler().fit_transform(df2)
#X_std

print(X_std)
len(X_std)


# # Covariance Matrix

# Variance reports variation of a single random variable — let’s say the weight of a person, and covariance reports how much two random variables vary.
# The diagonal elements of a coveriance matrix are identical and the matrix is symmetrical. 
# A covariance matrix is a square matrix that shows the covariance between many different variables. 
# A positive number for covariance indicates that two variables tend to increase or decrease in tandem. 
# A negative number for covariance indicates that as one variable increases, a second variable tends to decrease.
# Zero means no correlation.

# A covariance matrix is a more generalized form of a simple correlation matrix. Correlation is a scaled version of covariance; note that the two parameters always have the same sign (positive, negative, or 0).

# In[23]:


cov_matrix = np.cov(X_std.T)


# In[24]:


#Step 5: Compute covariance matrix. Represents how much two random variables vary.
#These values show the distribution magnitude and direction of multivariate data in a multidimensional space and 
#can allow you to gather information about how data spreads among two dimensions.
mean_vec = np.mean(X_std, axis=0)
cov_matrix = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
#print('Covariance matrix n%s' %cov_mat)
print(cov_matrix)


# In[27]:


mask = np.triu(np.ones_like(cov_matrix, dtype= int))
sns.heatmap(cov_matrix, annot= False, fmt='g')
plt.title('Covariance matrix showing correlation coefficients', color = "green", size = 10)
plt.show()


# Eigenvectors are simple unit vectors, and eigenvalues(or explained variance) are coefficients which give the magnitude to the eigenvectors.
# 

# In[32]:


#Step 6: Calculate eigenvectors and eigenvalues
#Calculating eigenvectors and eigenvalues on covariance matrix.
#eigenvalue function returns two type of arrays, one dimensional array representing the eigenvalues in 
#the position of the input and another two dimensional array giving the eigenvector corresponding to the columns in the input matrix.
cov_matrix = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
print('Eigenvectors %s' %eig_vecs)


# In[33]:


print('Eigenvalues %s' %eig_vals)


# In[36]:


#Step 7: Compute the feature vector.  rearrange the eigenvalues in descending order. 
#This represents the significance of the principal components in descending order:
# Visually confirm that the list is correctly sorted by decreasing eigenvalues
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])
#The top 10 eigenvalues adds up to 71% of the data. 


# In[40]:


#Step 8: Use the PCA() function to reduce the dimensionality of the data set
pca = PCA()
pca.fit_transform(df2)
print(pca.explained_variance_ratio_)
len(pca.explained_variance_ratio_) 


# In[41]:


#Step 9:Projecting the variance to the Principle Components
pca = PCA().fit(X_std)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.title('Scree plot of cumulative explained variance')
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.show()


# In[42]:


#Setting up PCA components to 10
pd.set_option("display.max_rows", None)
pca = PCA(n_components = 5)
pca_features = pca.fit_transform(X_std)
pca_features


# In[43]:


#Creating a PCA df to store the 10 PCs
pca_df = pd.DataFrame(
    data =pca_features, 
    columns=['PC1', 'PC2', 'PC3','PC4','PC5'])


# In[44]:


print('Shape before PCA: ', X_std.shape)
print('Shape after PCA: ', pca_features.shape)


# In[58]:


#The explained variance, or eigenvalue, in PCA shows the variance that can be attributed to
#each of the principal components. 
pca.explained_variance_


# In[59]:


#Bar Plot the explained variance
plt.bar(
    range(1,len(pca.explained_variance_)+1),
    pca.explained_variance_
    )
 
plt.xlabel('PCA Features')
plt.ylabel('Explained variance')
plt.title('PCA features and eigenvalues(explained variance)')
plt.show()


# In[60]:


from sklearn.linear_model import RidgeCV


# In[61]:


#It shows the first 5 PC values for each column. 
from sklearn import preprocessing
data_scaled = pd.DataFrame(preprocessing.scale(df2),columns = df2.columns) 
# PCA
pca = PCA(n_components=3)
pca.fit_transform(data_scaled)

# Dump components relations with features:
print(pd.DataFrame(pca.components_,columns=data_scaled.columns,index = ['PC-1','PC-2','PC3']))


# In[62]:


post_pca_array = pca.fit_transform(data_scaled)

print(data_scaled.shape)


# In[63]:


df.shape


# In[64]:


df.to_csv("RCdf-26Cols.csv")


# In[65]:


#df columns to cols variable.
cols = df.columns
cols


# # End of PCA

# In[66]:


#Create feature_names and target varaibles
feature_names = pd.DataFrame(df, columns=['Heading(mag)', 'Baro Setting Pilot', 'Ground Track Copilot',
       'Yaw Rate', 'Turn Rate', 'Flight Path Angle - VV-[0]',
       'Flight Path Angle - VV-[1]', 'Flight Path Angle - VV-[2]',
       'Ground Track - VV-[2]', 'Yaw Acceleration', 'Acceleration in Latitude',
       'Acceleration in Normal', 'Right Brake Pos', 'TOGA Status',
       'AP1 Status', 'GPS 1 NAV ID', 'NAV 2 NAV ID', 'NAV 2 DME Distance',
       'NAV 2 DME Speed', 'FMS Waypoints', 'Nav1 Ver Deviation',
       'Tail Rotor Chip Warning', 'Transmission Chip Warning',
       'Transmission Oil Temp Warning'])


# In[67]:


feature_names.shape


# In[68]:


feature_names.head()


# In[69]:


column_names = list(feature_names.columns)
print(column_names)


# In[70]:


target1 = df['Dynamic Rollover']
target1.head()
print(target1.isnull().sum())


# In[71]:


target2 = df['LOW-G']
target2.head()
print(target2.isnull().sum())


# # Techniques to calculate feature Importance
# #Data split in  train/test split for Feature importance
# #scale the predictors with StandardScaler class.

# In[73]:


#https://towardsdatascience.com/3-essential-ways-to-calculate-feature-importance-in-python-2f9149592155
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split #https://towardsdatascience.com/3-essential-ways-to-calculate-feature-importance-in-python-2f9149592155
#3 Essential Ways to Calculate Feature Importance in Python
#Dario Radečić.Follow Published in Towards Data Science ·6 min read·Jan 14, 2021
rcParams['figure.figsize'] = 14, 7
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False


# #df_coldrop is being used.
# #make a train/test split and scale the predictors with the StandardScaler class:
# #X is the predictors, y and z has the response variables.
# X = df.drop(['Dynamic Rollover', 'LOW-G'], axis=1)
# y = pd.DataFrame(df, columns=['Dynamic Rollover'])
# z = pd.DataFrame(df,columns=['LOW-G'])
# #adding z to test a 3d plot
# X_train, X_test, y_train, y_test,z_train,z_test = train_test_split(X, y,z, test_size=0.25, random_state=42)
# 
# ss = StandardScaler()
# #Scaling X 
# X_scaled = ss.fit_transform(X)
# #Scaling the train and test sets.
# X_train_scaled = ss.fit_transform(X_train)
# X_test_scaled = ss.transform(X_test)
# 
# 

# In[74]:


#make a train/test split and scale the predictors with the StandardScaler class:
#X is the predictors, y and z has the response variables.
X = feature_names
y = target1
z = target2
#adding z to test a 3d plot
X_train, X_test, y_train, y_test,z_train,z_test = train_test_split(X, y,z, test_size=0.25, random_state=42)

ss = StandardScaler()
#Scaling X 
X_scaled = ss.fit_transform(X)
#Scaling the train and test sets.
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)


# In[75]:


X.shape


# In[76]:


pca = PCA().fit(X_scaled)


# In[77]:


##D plot worked but I am not sure if it represents the data correctly. 
import numpy as np
import matplotlib.pyplot as plt
 
from mpl_toolkits import mplot3d
plt.style.use('default')
 
# Prepare 3D graph
fig = plt.figure()
ax = plt.axes(projection='3d')
 
# Plot scaled features
xdata = X_scaled[:,0]
ydata = X_scaled[:,1]
zdata = X_scaled[:,1]
 
# Plot 3D plot
ax.scatter3D(xdata, ydata, zdata, c=zdata , cmap='viridis')
# Plot title of graph
plt.title(f'3D Scatter of Rotorcraft data')

# Plot x, y, z even ticks
ticks = np.linspace(-3, 3, num=5)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_zticks(ticks)
 
# Plot x, y, z labels
ax.set_xlabel('Predictors', rotation=150)
ax.set_ylabel('Dynamic Rollover')
ax.set_zlabel('LOW-G',rotation = 60)
plt.show()


# In[78]:


plt.plot(pca.explained_variance_ratio_.cumsum(), lw=3, color='#087E8B')
plt.title('Cumulative explained variance by number of PCs', size=20)
plt.show()


# PCA loadings are the coefficients of the linear combination of the original variables from which the PCs are constructed.

# In[79]:


#PCA loadings. It is use to find corr between actual var and PCs.
loadings = pd.DataFrame(
    data=pca.components_.T * np.sqrt(pca.explained_variance_), 
    columns=[f'PC{i}' for i in range(1, len(X_train.columns) + 1)],
    index=X_train.columns
)
loadings.head()


# In[80]:


#Let’s visualize the correlations between all of the input features and the first principal components.
pc1_loadings = loadings.sort_values(by='PC1', ascending=False)[['PC1']]
pc1_loadings = pc1_loadings.reset_index()
pc1_loadings.columns = ['Attribute', 'CorrelationWithPC1']


# In[81]:


#Set up the show all the rows and columns. 
pd.set_option("display.max_rows", None)
print(pc1_loadings)


# In[82]:


plt.axis([0,25, 0, 10])
plt.bar(x=pc1_loadings['Attribute'], height=pc1_loadings['CorrelationWithPC1'], color='#087E8B')
plt.title('Feature importance using PCA loading scores', size=20)
plt.xticks(rotation='vertical')
plt.show()


# # Regression Feature importances obtained from coefficients

# In[83]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train_scaled, y_train)
LRimportances = pd.DataFrame(data={
    'Attribute': X_train.columns,
    'Importance': model.coef_[0]
})


# In[84]:


LRimportances = LRimportances.sort_values(by= 'Importance', ascending=False)
LRimportances 


# In[85]:


plt.axis([0,25, 0, 15])
plt.bar(x= LRimportances['Attribute'], height= LRimportances['Importance'], color='#087E8B')
plt.title('Feature importance using coefficients for Regression model', size=20)
plt.xticks(rotation='vertical')
plt.show()


# # Classifier feature importance

# In[86]:


#https://towardsdatascience.com/3-essential-ways-to-calculate-feature-importance-in-python-2f9149592155
#from sklearn.linear_model import LogisticRegression. It did not work due to the label field.
import xgboost
from xgboost import XGBClassifier


# In[87]:


model = XGBClassifier()
model.fit(X_train_scaled, y_train)
XGBimportances = pd.DataFrame(data={
    'Attribute': X_train.columns,
    'Importance': model.feature_importances_
})
XGBimportances = XGBimportances.sort_values(by='Importance', ascending=False)


# In[88]:


plt.axis([0,30, 0,1])
#'Feature importance using thecoefficients of the train model for a classifier model
plt.bar(x=XGBimportances['Attribute'], height= XGBimportances['Importance'], color='#087E8B')
plt.title('Feature importance using Classifier technique', size=20)
plt.xticks(rotation='vertical')
plt.show()


# In[89]:


pd.set_option("display.max_rows", None)
XGBimportances = XGBimportances.sort_values(by='Importance', ascending=False)
XGBimportances


# # End of feature importance

# In[ ]:




