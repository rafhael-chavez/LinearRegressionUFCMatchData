import numpy as np
import pandas as pd
import math
import re
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Cargar los datasets
fighters_df = pd.read_csv('ufc_fighters.csv')
events_df = pd.read_csv('ufc_event_data.csv')

# Funciones para la conversion de alturas, pesos y alcance
def height_to_inches(height_str):
    match = re.match(r"(\d+)' (\d+)", height_str)
    if match:
        feet = int(match.group(1))
        inches = int(match.group(2))
        return feet * 12 + inches
    return np.nan

def weight_to_float(weight_str):
    match = re.match(r"(\d+\.?\d*) lbs\.", weight_str)
    if match:
        return float(match.group(1))
    return np.nan   

def reach_to_float(reach_str):
    if pd.isna(reach_str) or reach_str == '--':
        return np.nan
    match = re.match(r"(\d+\.?\d*)\"", reach_str)
    if match:
        return float(match.group(1))
    return np.nan

# Limpiar y convertir las columnas del DataFrame de luchadores
fighters_df['Height'] = fighters_df['Height'].apply(height_to_inches)
fighters_df['Weight'] = fighters_df['Weight'].apply(weight_to_float)
fighters_df['Reach'] = fighters_df['Reach'].apply(reach_to_float)

# Manejar valores nulos
fighters_df.dropna(subset=['Height', 'Weight', 'Reach'], inplace=True)

# Crear una nueva columna con el nombre completo
fighters_df['Full Name'] = fighters_df['First Name'] + ' ' + fighters_df['Last Name']

# Unir los datasets usando la columna 'Full Name'
merged_df = events_df.merge(fighters_df, left_on='Fighter1', right_on='Full Name', how='left')
merged_df = merged_df.merge(fighters_df, left_on='Fighter2', right_on='Full Name', suffixes=('_F1', '_F2'), how='left')

# Eliminar columnas innecesarias
merged_df.drop(columns=['First Name_F1', 'Last Name_F1', 'Full Name_F1', 'First Name_F2', 'Last Name_F2', 'Full Name_F2'], inplace=True)

# Manejar valores nulos
merged_df.fillna(0, inplace=True)

# Seleccionar las caracteristicas relevantes y la etiqueta
X = merged_df[['Height_F1', 'Weight_F1', 'Reach_F1', 'Wins_F1', 'Losses_F1', 'Draws_F1',
               'Height_F2', 'Weight_F2', 'Reach_F2', 'Wins_F2', 'Losses_F2', 'Draws_F2']]
y = (merged_df['Fighter1_Fighter2'] == merged_df['Fighter1']).astype(int) 

# Escalar las caracteristicas
sc = StandardScaler()
X = sc.fit_transform(X)

# Dividir los datos en conjuntos de entrenamiento y prueba 
test_size = 0.3
random_state = 42

# Fijar la semilla para reproducibilidad
np.random.seed(random_state)

# Obtener el numero total de ejemplos
num_samples = len(X)

# Generar indices aleatorios para los datos
indices = np.arange(num_samples)
np.random.shuffle(indices)

# Calcular el indice de separacion
split_index = int(num_samples * (1 - test_size))

# Dividir los indices en entrenamiento y prueba
train_indices = indices[:split_index]
test_indices = indices[split_index:]

# Crear los conjuntos de entrenamiento y prueba
X_train = X[train_indices]
X_test = X[test_indices]
y_train = y[train_indices]
y_test = y[test_indices]

# Inicializar theta
theta = np.random.randn(len(X_train[0]) + 1, 1)

# Anadir columna de 1s a X_train (para el termino de sesgo)
X_train_vect = np.c_[np.ones((len(X_train), 1)), X_train]

# Funcion sigmoide
def sigmoid_function(X):
    return 1 / (1 + np.exp(-X))

# Implementacion de la regresion logistica
def log_regression(X, y, theta, alpha, epochs):
    y_ = np.reshape(y, (len(y), 1))  # reshape del vector y para tener dimensiones (n_samples, 1)
    N = len(X)  # numero de muestras en X
    avg_loss_list = []  # lista para almacenar la perdida promedio en cada epoch
    for epoch in range(epochs):  # bucle que recorre cada epoch
        sigmoid_x_theta = sigmoid_function(X.dot(theta))  # calcula la hipotesis usando la funcion sigmoide
        grad = (1/N) * X.T.dot(sigmoid_x_theta - y_)  # calcula el gradiente para ajustar los parametros theta
        theta = theta - (alpha * grad)  # actualiza los parametros theta usando la tasa de aprendizaje alpha
        hyp = sigmoid_function(X.dot(theta))  # recalcula la hipotesis despues de actualizar theta
        avg_loss = -np.sum(np.dot(y_.T, np.log(hyp)) + np.dot((1-y_).T, np.log(1-hyp))) / len(hyp)  # calcula la perdida promedio
        if epoch % 1000 == 0:  # imprime la perdida cada 100 epochs
            print(f'epoch: {epoch} | avg_loss: {avg_loss}')
        avg_loss_list.append(avg_loss)  # anade la perdida promedio a la lista
    # plt.plot(np.arange(1, epochs), avg_loss_list[1:], color='red')  # grafico de la funcion de costo
    # plt.title('Funcion de Costo')  # titulo del grafico
    # plt.xlabel('Epochs')  # etiqueta para el eje x
    # plt.ylabel('Costo')  # etiqueta para el eje y
    # plt.show()  # muestra el grafico
    return theta  # devuelve los parametros theta optimizados

# Entrenar el modelo
epochs = 10000
alpha = 0.01
theta = log_regression(X_train_vect, y_train, theta, alpha, epochs)

# Hacer predicciones en el conjunto de prueba
X_test_vect = np.c_[np.ones((len(X_test), 1)), X_test]  # anadir una columna de unos a X_test para el termino independiente
y_pred = sigmoid_function(X_test_vect.dot(theta)) >= 0.5  # calcular la hipotesis y convertir a 1 si es mayor o igual a 0.5
y_pred = y_pred.astype(int)  # convertir las predicciones booleanas a enteros (0 o 1)


# Mostrar la matriz de confusion
cm = confusion_matrix(y_test, y_pred)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Matriz de Confusion')
# plt.show()

# Mostrar el reporte de clasificacion
print(classification_report(y_test, y_pred))

# Calcular la precision
accuracy = accuracy_score(y_test, y_pred)
print(f"Porcentaje de aciertos: {accuracy * 100:.2f}%")
