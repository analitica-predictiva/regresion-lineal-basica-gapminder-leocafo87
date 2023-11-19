"""
Regresión Lineal Univariada
-----------------------------------------------------------------------------------------

En este laboratio se construirá un modelo de regresión lineal univariado.

"""
import numpy as np
import pandas as pd


def pregunta_01():
    df = pd.read_csv('gm_2008_region.csv')

    y = df['life'].copy()
    X = df['fertility'].copy()

    print(y.shape)

    print(X.shape)

    y_reshaped = y.values.reshape(y.shape[0], 1)

    X_reshaped = X.values.reshape(X.shape[0], 1)

    print(y_reshaped.shape)

    print(X_reshaped.shape)


def pregunta_02():

    df = pd.read_csv('gm_2008_region.csv')

    print(df.shape)

    print("{:.4f}".format(df['life'].corr(df['fertility'])))

    print("{:.4f}".format(df['life'].mean()))

    print(df['fertility'].__class__)

    print("{:.4f}".format(df['GDP'].corr(df['life'])))


def pregunta_03():
    df = pd.read_csv('gm_2008_region.csv')

    X_fertility = df['fertility'].copy()

    y_life = df['life'].copy()

    from sklearn.linear_model import LinearRegression

    reg = LinearRegression()

    prediction_space = np.linspace(min(X_fertility),max(X_fertility),).reshape(-1, 1)

    reg.fit(X_fertility.values.reshape(-1, 1), y_life.values.reshape(-1, 1))

    y_pred = reg.predict(prediction_space)

    print(reg.score(X_fertility.values.reshape(-1, 1), y_life.values.reshape(-1, 1)).round(4))


def pregunta_04():
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    df = pd.read_csv('gm_2008_region.csv')

    X_fertility = df['fertility'].copy()

    y_life = df['life'].copy()

    (X_train, X_test, y_train, y_test,) = train_test_split(X_fertility,y_life,test_size = 0.20,random_state = 53 )

    linearRegression = LinearRegression()

    linearRegression.fit(X_train.values.reshape(-1, 1), y_train.values.reshape(-1, 1))

    y_pred = linearRegression.predict(X_test.values.reshape(-1, 1))

    print("R^2: {:6.4f}".format(linearRegression.score(X_test.values.reshape(-1, 1), y_test.values.reshape(-1, 1))))
    rmse = mean_squared_error(y_test, y_pred, squared = False)
    print("Root Mean Squared Error: {:6.4f}".format(rmse))
