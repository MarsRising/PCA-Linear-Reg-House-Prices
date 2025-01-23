#!/usr/bin/env python
# coding: utf-8

# In[484]:


#Import necessary libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import chi2_contingency, ttest_ind, f_oneway
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
print("libs downloaded!")


# In[485]:


#define path
file_path = 'C:/Users/tyler/OneDrive - SNHU/WGU/Statistical Data Mining/D600 Task 3 Dataset 1 Housing Information.csv'
#import excel to dataframe
df=pd.read_csv(file_path)
print(df.head)


# In[486]:


#drop irrelevatn columns
df.drop(columns=["ID"], inplace=True)
df.info()


# In[487]:


df.describe(include='all')


# In[488]:


#use jsut the numeric variables to find significinat continuous variables
numeric_df = df.select_dtypes(include=['number'])
correlation_matrix=numeric_df.corr()
print(correlation_matrix)


# In[489]:


#get rid of insignificant correlations
df = df.drop(columns=['BackyardSpace', 'CrimeRate', 'SchoolRating', 'AgeOfHome', 'DistanceToCityCenter', 'EmploymentRate', 'PropertyTaxRate', 'LocalAmenities', 'TransportAccess', 'Floors', 'Windows'])
df.info()


# In[490]:


#xheck pearson correlation jsut for the significant ones
numeric_df = df.select_dtypes(include=['number'])
correlation_matrix=numeric_df.corr()
print(correlation_matrix)


# In[491]:


#t-test for binary variables 
fireplace_yes = df[df['Fireplace'] == 'Yes']['Price']
fireplace_no = df[df['Fireplace'] == 'No']['Price']
Garage_yes = df[df['Garage'] == 'Yes']['Price']
Garage_no = df[df['Garage'] == 'No']['Price']
IsLuxury_yes = df[df['IsLuxury'] == 1]['Price']
IsLuxury_no = df[df['IsLuxury'] == 0]['Price']

t_stat, p_value = ttest_ind(fireplace_yes, fireplace_no)
print(f"T-test p-value for 'Fireplace' and 'Price': {p_value}")
t_stat, p_value = ttest_ind(Garage_yes, Garage_no)
print(f"T-test p-value for 'Garage' and 'Price': {p_value}")
t_stat, p_value = ttest_ind(IsLuxury_yes, IsLuxury_no)
print(f"T-test p-value for 'IsLuxury' and 'Price': {p_value}")


# In[492]:


#ANOVA due to multiple categorical values in house colors
house_colors = [df[df['HouseColor'] == color]['Price'] for color in df['HouseColor'].unique()]
f_stat, p_value = f_oneway(*house_colors)
print(f"ANOVA p-value for 'HouseColor' and 'Price': {p_value}")


# In[493]:


#Out of the 4 categorical values the only 2 significant variables are Fireplace and IsLuxury DROP the other two
df.drop(columns=['HouseColor', 'Garage'], inplace=True)
df.info()


# In[494]:


#Now I am down to simply the variables that have the most significance.
#Before moving forward Variance Inflation Factor is important to handle milticollinearity
#VIF=1 NO CORRELATION     VIF>1 CORRELATED BUT NOT TOO STRONG   VIF>5 HIGH MULTICOLLINEARITY
cont_vars = df[['SquareFootage', 'NumBathrooms', 'NumBedrooms', 'RenovationQuality', 'PreviousSalePrice']]
#constraint
X = add_constant(cont_vars)
#Calculate VIF
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)


# In[495]:


#ALL VARIABLES ARE GOOD! No Multicollinearity
#check isluxury is set up accurately
print(df.dtypes)
#was not. Update it to categorical
df['IsLuxury'] = df['IsLuxury'].astype('category')
#DESCRIPTIVE STATISTICS
df.describe(include='all')


# In[496]:


#Univariate analysis Price
sns.histplot(df['Price'], kde=False, bins=20, color='skyblue')
plt.title('Distribution of Price')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()
#boxplot
sns.boxplot(x=df['Price'], color='lightgreen')
plt.title('Boxplot of Price')
plt.xlabel('Price')
plt.show()


# In[497]:


#SquareFootage
#Histogram
sns.histplot(df['SquareFootage'], kde=False, bins=20, color='skyblue')
plt.title('Distribution of Square Footage')
plt.xlabel('Square Footage')
plt.ylabel('Frequency')
plt.show()
#boxplot
sns.boxplot(x=df['SquareFootage'], color='lightgreen')
plt.title('Boxplot of Square Footage')
plt.xlabel('Square Footage')
plt.show()


# In[498]:


#NumBathrooms
#Histogram
sns.histplot(df['NumBathrooms'], kde=False, bins=20, color='skyblue')
plt.title('Distribution of Number of Bathrooms')
plt.xlabel('Number of Bathrooms')
plt.ylabel('Frequency')
plt.show()
#boxplot
sns.boxplot(x=df['NumBathrooms'], color='lightgreen')
plt.title('Boxplot of NumBathrooms')
plt.xlabel('Number of Bathrooms')
plt.show()


# In[499]:


#NumBedrooms
#Histogram
sns.histplot(df['NumBedrooms'], kde=False, bins=20, color='skyblue')
plt.title('Distribution of Number of Bedrooms')
plt.xlabel('Number of Bedrooms')
plt.ylabel('Frequency')
plt.show()
#boxplot
sns.boxplot(x=df['NumBedrooms'], color='lightgreen')
plt.title('Boxplot of NumBedrooms')
plt.xlabel('Number of Bedrooms')
plt.show()


# In[500]:


#RenovationQuality
#Histogram
sns.histplot(df['RenovationQuality'], kde=False, bins=20, color='skyblue')
plt.title('Distribution of RenovationQuality')
plt.xlabel('RenovationQuality')
plt.ylabel('Frequency')
plt.show()
#boxplot
sns.boxplot(x=df['RenovationQuality'], color='lightgreen')
plt.title('Boxplot of RenovationQuality')
plt.xlabel('RenovationQuality')
plt.show()


# In[501]:


#Fireplace CATEGORICAL
sns.countplot(x='Fireplace', data=df)
plt.title('Fireplace Distribution')
plt.xlabel('Fireplace')
plt.ylabel('Count')
plt.show()


# In[502]:


#PreviousSalePrice
#Histogram
sns.histplot(df['PreviousSalePrice'], kde=False, bins=20, color='skyblue')
plt.title('Distribution of PreviousSalePrice')
plt.xlabel('PreviousSalePrice')
plt.ylabel('Frequency')
plt.show()
#boxplot
sns.boxplot(x=df['PreviousSalePrice'], color='lightgreen')
plt.title('Boxplot of PreviousSalePrice')
plt.xlabel('PreviousSalePrice')
plt.show()


# In[503]:


#IsLuxury CATEGORICAL
sns.countplot(x='IsLuxury', data=df)
plt.title('IsLuxury Distribution')
plt.xlabel('IsLuxury')
plt.ylabel('Count')
plt.show()


# In[504]:


continuous_vars = ['SquareFootage', 'NumBathrooms', 'NumBedrooms', 'RenovationQuality', 'PreviousSalePrice']

# Create scatter plots for each continuous variable against Price
for var in continuous_vars:
    plt.figure(figsize=(6, 4))
    sns.lmplot(x=var, y='Price', data=df)
    sns.scatterplot(x=df[var], y=df['Price'])
    plt.title(f'Scatterplot of {var} vs Price')
    plt.xlabel(var)
    plt.ylabel('Price')
    plt.show()


# In[505]:


#Bivariate for my categoricals 
sns.boxplot(x='Fireplace', y='Price', data=df)
plt.title('Boxplot of Price by Fireplace')
plt.xlabel('Fireplace')
plt.ylabel('Price')
plt.show()
sns.boxplot(x='IsLuxury', y='Price', data=df)
plt.title('Boxplot of Price by IsLuxury')
plt.xlabel('IsLuxury')
plt.ylabel('Price')
plt.show()


# In[506]:


#The boxes in the Fireplac, as well as Medians, are very close.
#I will remove Fireplace as this may not be necessarily as strong as first expected, and I would like to not complicate my model
df.drop(columns=["Fireplace"], inplace=True)
df.info()


# In[ ]:




