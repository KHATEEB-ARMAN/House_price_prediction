import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
house_price_dataset=sklearn.datasets.fetch_california_housing()
house_price_dataset
house_price_dataframe=pd.DataFrame(data=house_price_dataset.data,columns=house_price_dataset.feature_names)
house_price_dataframe.head()
house_price_dataframe['price']=house_price_dataset.target
house_price_dataframe.head()
# checking the number of rows and columns in the dataframe
house_price_dataframe.shape
# check for missing values
house_price_dataframe.count()
house_price_dataframe.isnull().any()
# statistical measures of the dataset
house_price_dataframe.describe()
correlation = house_price_dataframe.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
X = house_price_dataframe.drop(['price'], axis=1)
Y = house_price_dataframe['price']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(X.shape, X_train.shape, X_test.shape)
model = XGBRegressor()
#training the model with X_train
model.fit(X_train, Y_train)
# accuracy for prediction on training data
training_data_prediction = model.predict(X_train)
print(training_data_prediction)
# R Squared Error
score_1 = metrics.r2_score(Y_train, training_data_prediction)

# Mean Absolute Error
score_2 = metrics.mean_absolute_error(Y_train, training_data_prediction)

print('R Sqaured Error:', score_1)
print('Mean Absolute Error:', score_2)

plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Price vs Predicted Price")
plt.show()
# accuracy for prediction on test data
test_data_prediction = model.predict(X_test)

# R Squared Error
score_1 = metrics.r2_score(Y_test, test_data_prediction)

# R Squared Error
score_1 = metrics.r2_score(Y_test, test_data_prediction)

# Mean Absolute Error
score_2 = metrics.mean_absolute_error(Y_test, test_data_prediction)

print('R Sqaured Error:', score_1)
print('Mean Absolute Error:', score_2)