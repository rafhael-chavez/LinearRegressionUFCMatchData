# LinearRegressionUFCMatchData
# UFC Fight Prediction Using Logistic Regression

Este proyecto utiliza un modelo de regresión logística para predecir el resultado de las peleas de UFC basado en datos de luchadores y eventos.

## Contenido del Proyecto

- **`ufc_fighters.csv`**: Datos de los luchadores de UFC, incluyendo atributos como altura, peso, alcance, y estadísticas de victorias, derrotas y empates.
- **`ufc_event_data.csv`**: Datos de eventos de UFC, incluyendo información sobre los combates y resultados.
- **`ufc_regresion_logistica.py`**: Script principal que realiza el preprocesamiento de datos, entrena un modelo de regresión logística y evalúa el rendimiento del modelo.

## Requisitos

Asegúrate de tener instalados los siguientes paquetes en tu entorno de Python:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

Puedes instalar los paquetes necesarios usando `pip`:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn

### Entrenamiento del Modelo

1. **Entrenamiento**: El script entrena un modelo de regresión logística con los datos procesados.
2. **Parámetros de Entrenamiento**: El modelo se entrena durante 10,000 épocas con una tasa de aprendizaje de 0.01.

### Evaluación del Modelo

1. **Predicciones**: Se realizan predicciones sobre un conjunto de prueba.
2. **Matriz de Confusión y Reporte de Clasificación**: Se muestra la matriz de confusión y el reporte de clasificación.
3. **Precisión**: Se calcula y muestra el porcentaje de precisión del modelo.

Para ejecutar el script, simplemente corre:

```bash
python ufc_regresion_logistica.py


## Comparación con Investigación Previa

En un estudio relacionado titulado ["Predicting UFC Fight Outcomes with Machine Learning"]([enlace-al-estudio](https://kth.diva-portal.org/smash/get/diva2:1878726/FULLTEXT01.pdf), se logró un porcentaje de aciertos del 60% utilizando un modelo similar pero con un conjunto de datos diferente. En comparación, nuestro modelo alcanzó una precisión del 51%.

