import numpy as np
import pandas as pd

train=pd.read_csv('C:/Users/Abhiijit/Desktop/train.csv')
test=pd.read_csv('C:/Users/Abhiijit/Desktop/test.csv')

print('train data shape',train.shape)
print('test data shape',test.shape)

train.head()


import matplotlib.pyplot as plt

plt.style.use(style='ggplot')
plt.rcParams['figure.figsize']= (10,6)

train.SalePrice.describe()

print('skew is:', train.SalePrice.skew())
plt.hist(train.SalePrice,color='blue')
plt.show()

target=np.log(train.SalePrice)
print("skew is:",target.skew())
plt.hist(target,color='blue')
plt.show()

numeric_features=train.select_dtypes(include=[np.number])
numeric_features.dtypes

corr=numeric_features.corr()
print(corr['SalePrice'].sort_values(ascending=False)[:5],'\n')
print(corr['SalePrice'].sort_values(ascending=False)[-5:])

#the first 5 features are most positively correlated
#we can use the .unique() method to get the unique values

train.OverallQual.unique()

# we set index='OverAllQual' and values='SalePrice.we chose to look at the median here.

quality_pivot=train.pivot_table(index='OverallQual',values='SalePrice',aggfunc=np.median)
quality_pivot

quality_pivot.plot(kind='bar',color='blue')
plt.xlabel('Overall quality')
plt.ylabel('median sale price')
plt.xticks(rotation=0)
plt.show()

#visualize the relationship between ground living area and sale price

plt.scatter(x=train['GrLivArea'],y=target)
plt.ylabel('sale price')
plt.xlabel('above ground living area square feet')
plt.show()

# now visualize for garage area

plt.scatter(x=train['GarageArea'],y=target)
plt.ylabel('sale price')
plt.xlabel('garage area')
plt.show()

# we note that there are many homes with 0 for garage area, indicating that they dont have a garage.
# we will create a new dataframw with outliers removed

train=train[train['GarageArea']<1200]

plt.scatter(x=train['GarageArea'],y=np.log(train.SalePrice))
plt.xlim(-200,1600)
plt.ylabel('sale price')
plt.xlabel('garage area')
plt.show()

#handling null values

nulls=pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns=['Null Count']
nulls.index.name='Feature'
nulls

# we will use the unique method to return a list of unique values

print('unique values are',train.MiscFeature.unique())

categoricals=train.select_dtypes(exclude=[np.number])
categoricals.describe()

print('original: \n')
print(train.Street.value_counts(),'\n')

#our model needs numerical data so we will do one-hot encoding.

train['enc_street']=pd.get_dummies(train.Street,drop_first=True)
test['enc_street']=pd.get_dummies(train.Street,drop_first=True)

print('encoded \n')
print(train.enc_street.value_counts())

# we will engineer another feature.

condition_pivot=train.pivot_table(index='SaleCondition',values='SalePrice',aggfunc=np.median)
condition_pivot.plot(kind='bar',color='blue')
plt.xlabel('sale condition')
plt.ylabel('median sale price')
plt.xticks(rotation=0)
plt.show()

def encode(x):
 return 1 if x == 'Partial' else 0
train['enc_condition'] = train.SaleCondition.apply(encode)
test['enc_condition'] = test.SaleCondition.apply(encode)

condition_pivot = train.pivot_table(index='enc_condition', values='SalePrice', aggfunc=np.median)
condition_pivot.plot(kind='bar', color='blue')
plt.xlabel('Encoded Sale Condition')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()

data=train.select_dtypes(include=[np.number]).interpolate().dropna()
sum(data.isnull().sum() != 0)

#data.drop will say which features to exclude
y=np.log(train.SalePrice)
X=data.drop(['SalePrice','Id'],axis=1)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.33)

from sklearn import linear_model

lr=linear_model.LinearRegression()

model=lr.fit(X_train,y_train)


print('R^2: is',model.score(X_test,y_test))
predictions=model.predict(X_test)

from sklearn.metrics import mean_squared_error

print("mean_squared_error",mean_squared_error(y_test,predictions))
actual_values = y_test
plt.scatter(predictions, actual_values, alpha=.7,
            color='b') #alpha helps to show overlapping data
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.title('Linear Regression Model')
plt.show()

