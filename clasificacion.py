# Ejemplo de clasificacion
# Predecir los ingresos de una persona en funcion de sus caracteristicas
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# importar archivo
data = pd.read_csv('original.csv')
#print(data)

#seleccionar una columna
ingresos = data['income'].unique()

# cambiar los valores de income a 0 o 1
def cambio_valor(valor):
    if valor == '<=50K':
        return 0
    else:
        return 1
#aplicar valores
data['income'] = data['income'].apply(cambio_valor)

#eliminar columna income
datos_x = data.drop('income', axis=1)

#seleccionar datos de la columna income
datos_y = data['income']

#dividir datos de Entrenamiento y test
x_train, x_test, y_train , y_test = train_test_split(datos_x, datos_y,test_size=0.3)

# obtner datos de las columnas
gender = tf.feature_column.categorical_column_with_vocabulary_list("gender",['Female, Male'])
occupation = tf.feature_column.categorical_column_with_hash_bucket("occupation",hash_bucket_size=1000)
marital_status = tf.feature_column.categorical_column_with_hash_bucket("marital-status",hash_bucket_size=1000)
relationship = tf.feature_column.categorical_column_with_hash_bucket("relationship",hash_bucket_size=1000)
education = tf.feature_column.categorical_column_with_hash_bucket("education",hash_bucket_size=1000)
native_country = tf.feature_column.categorical_column_with_hash_bucket("native-country",hash_bucket_size=1000)
workclass = tf.feature_column.categorical_column_with_hash_bucket("workclass",hash_bucket_size=1000)

age = tf.feature_column.numeric_column("age")
fnlwgt = tf.feature_column.numeric_column("fnlwgt")
educational_num = tf.feature_column.numeric_column("educational-num")
capital_gain = tf.feature_column.numeric_column("capital-gain")
capital_loss = tf.feature_column.numeric_column("capital-loss")
hours_per_week = tf.feature_column.numeric_column("hours-per-week")

# crear lista
columnas_categorias = [gender, occupation, marital_status, relationship, education, native_country, workclass, age, fnlwgt, educational_num, capital_gain, capital_loss, hours_per_week ]

#funcion de entrada para estimar
funcion_entrada = tf.estimator.inputs.pandas_input_fn(x=x_train, y=y_train, batch_size=100, num_epochs=None, shuffle=True)
modelo = tf.estimator.LinearClassifier(feature_columns=columnas_categorias)
#entrenar modelo
modelo.train(input_fn=funcion_entrada, steps=8000)
funcion_pred = tf.estimator.inputs.pandas_input_fn(x=x_test, batch_size=len(x_test), shuffle=False)
#generador de predicciones
generador_pred = modelo.predict(input_fn = funcion_pred)
#lista de predicciones
predicciones = list(generador_pred)
predict_fin = [prediccion['class_ids'][0] for prediccion in predicciones]
print(classification_report(y_test, predict_fin))
