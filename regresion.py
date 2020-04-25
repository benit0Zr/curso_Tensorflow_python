#Ejemplo de regresion con tensorflow
#Estimar el valor medio de las casas

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Leer archivo
casas = pd.read_csv('precios_casas.csv')

datos_x = casas.drop('median_house_value', axis=1)

#valores de la columna median_house_vaue
datos_y = casas['median_house_value']
#dividir datos 70% para train y 30% para test
x_train, x_test, y_train, y_test = train_test_split(datos_x, datos_y, test_size=0.3)
#Normalizar los datos(0 a 1)
normalizador = MinMaxScaler()
normalizador.fit(x_train)
MinMaxScaler(copy=True, feature_range=(0,1))
x_train = pd.DataFrame(data=normalizador.transform(x_train), columns=x_train.columns, index=x_train.index)
x_test = pd.DataFrame(data=normalizador.transform(x_test), columns=x_test.columns, index=x_test.index)
#crear varibles de las columnas
longitude = tf.feature_column.numeric_column('longitude')
latitude = tf.feature_column.numeric_column('latitude')
housing_median_age = tf.feature_column.numeric_column('housing_median_age')
total_rooms = tf.feature_column.numeric_column('total_rooms')
total_bedrooms = tf.feature_column.numeric_column('total_bedrooms')
population = tf.feature_column.numeric_column('population')
households = tf.feature_column.numeric_column('households')
median_income = tf.feature_column.numeric_column('median_income')
#lista de columnas
columnas = [longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income]
#funcion de funcion_entrada
funcion_entrada = tf.estimator.inputs.pandas_input_fn(x=x_train, y=y_train, batch_size=10, num_epochs=1000, shuffle=True)
#Crear modelo
modelo = tf.estimator.DNNRegressor(hidden_units=[10,10,10], feature_columns=columnas)
#entrenar modelo
modelo.train(input_fn=funcion_entrada, steps=8000)
#generar predicciones
funcion_entrada_predic = tf.estimator.inputs.pandas_input_fn(x=x_test, batch_size=10, num_epochs=1, shuffle=False)
generador_pred = modelo.predict(funcion_entrada_predic)
predicciones = list(generador_pred)
#almacenar las predicciones
predic_finales = []
for prediccion in predicciones:
    predic_finales.append(prediccion['predictions'])
#Error medio de la estimacion
error_md = mean_squared_error(y_test, predic_finales)**0.5
print(error_md)
