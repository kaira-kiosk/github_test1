import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score

data = {
    'Temperature': [10, 15, 20, 25, 30, 35, 40],
    'Sales': [20, 35, 65, 110, 180, 270, 390]
}
df=pd.DataFrame(data)
print(df)
print(df.info())

x=df[['Temperature']]
y=df["Sales"]
print("features:")
print(x)
print("target:")
print(y)
plt.scatter(x, y)

plt.xlabel('Temperature')
plt.ylabel('Sales')

plt.title('Temperature vs Ice Cream Sales')

plt.show()

x_train,x_test,y_train,y_test=train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)
print("x_train:")
print(x_train)
print("x_test")
print(x_test)
print("trainingsamples:",len(x_train))
print("testingsamples:",len(x_test))
poly=PolynomialFeatures(degree=2)
x_train_poly=poly.fit_transform(x_train)
x_test_poly=poly.transform(x_test)
print("original x :")
print(x_train)
print("polynomial x :")
print(x_train_poly)

model= LinearRegression()
model.fit(x_train_poly,y_train)

print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

y_pred=model.predict(x_test_poly)

print("Actual Values:", y_test.values)
print("Predicted Values:", y_pred)



mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("mse:",mse)
print("r2score:",r2)

X_plot = np.linspace(
    X_train.min().iloc[0],
    X_train.max().iloc[0],
    100
).reshape(-1, 1)

X_plot_poly = poly.transform(X_plot)

y_plot = model.predict(X_plot_poly)

plt.scatter(X, y, label='Actual Data')

plt.plot(X_plot, y_plot, color='red', label='Polynomial Curve')

plt.xlabel('Temperature')
plt.ylabel('Sales')

plt.title('Polynomial Regression')

plt.legend()

plt.show()
