"""
Student's Performance

Step 1 - Importing required libraries
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

"""
Step 2 - Importing the dataset
"""

link = "http://bit.ly/w-data"
data = pd.read_csv(link)
data.head(5)

"""
Step 3 - Visualizing the data
"""

data.plot(x='Hours', y='Scores', c='red', style='o')

plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()

"""
Step 4 - Performing data preprocessing
"""

X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0)

"""
Step 5 - Fit the dataset into Simple linear regression model
"""

from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train)

line = (regressor.coef_*X)+regressor.intercept_

"""
Step 6 - Visualising the correlation in the dataset
"""

plt.scatter(X, y, c='red', label='Scores')
plt.plot(X, line, c='black', label='Linear Regression')

plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.legend()
plt.show()

print(X_test)
y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df

from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred))


print('Coefficient of determination:', metrics.r2_score(y_test,y_pred))

"""
Step 7 - Predicting the value for dependent variable (Scores)
"""

hours = 9.25
own_pred = regressor.predict([[hours]])
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))
