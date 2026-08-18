# Ejercicio de clase — Regularización vs. KNN Regressor

## Contexto

En el notebook de ejemplo (`notebooks/Regularizacion_Ridge_Lasso_ElasticNet.ipynb`) vimos cómo Ridge, Lasso y Elastic Net modifican el vector de pesos de una regresión lineal. En este ejercicio ustedes van a extender ese análisis agregando un modelo **no paramétrico** (KNN Regressor) y decidiendo, con evidencia cuantitativa, **cuál de todos los modelos es el más robusto** para este problema.

## Dataset

Trabajarán con el dataset de Kaggle **[Home Value Insights](https://www.kaggle.com/datasets/prokshitha/home-value-insights)**, ya descargado en:

```
data/house_price_regression_dataset.csv
```

Columnas:

| Columna | Descripción |
|---|---|
| `Square_Footage` | Área construida en pies cuadrados |
| `Num_Bedrooms` | Número de habitaciones |
| `Num_Bathrooms` | Número de baños |
| `Year_Built` | Año de construcción |
| `Lot_Size` | Tamaño del lote (acres) |
| `Garage_Size` | Número de espacios de garaje |
| `Neighborhood_Quality` | Puntaje de calidad del vecindario (1–10) |
| `House_Price` | **Variable objetivo** — precio de la vivienda |

## Objetivos de aprendizaje

- Construir un pipeline de preprocesamiento reutilizable con `ColumnTransformer` (escalamiento numérico y, si aplica, codificación categórica).
- Entrenar y comparar modelos de **regresión lineal regularizada** (Ridge, Lasso, Elastic Net) contra un modelo **basado en distancias** (KNN Regressor).
- Seleccionar hiperparámetros usando *cross-validation* sobre el conjunto de entrenamiento (sin usar un conjunto de validación aparte).
- Evaluar y comparar modelos usando **MAE** y **R²**.
- Argumentar, con evidencia, cuál modelo es el más **robusto** — no solo cuál tiene el mejor número en una sola corrida.

## Instrucciones

### 1. Carga y exploración

- Cargar el CSV y revisar tipos de datos, valores nulos y estadísticas descriptivas.
- Identificar cuáles columnas son numéricas y cuáles serían categóricas (aunque en este dataset todas son numéricas, el pipeline debe construirse de forma general, detectando los tipos automáticamente).

### 2. Partición de datos

- Usar **únicamente** `train_test_split` para separar **train** y **test** (80/20, `random_state` fijo para reproducibilidad). **No usar un conjunto de validación aparte**: la selección de hiperparámetros debe hacerse con *cross-validation* sobre `train` (por ejemplo, `GridSearchCV`, `RidgeCV`, `LassoCV`, `ElasticNetCV`).

### 3. Pipeline de preprocesamiento

- Construir un `ColumnTransformer` dentro de un `Pipeline` que aplique `StandardScaler` a las variables numéricas y `OneHotEncoder` a las categóricas (si las hubiera).
- Todo modelo debe entrenarse **a través del pipeline**, nunca sobre los datos crudos ni sobre arreglos transformados manualmente fuera de él.

### 4. Modelos a entrenar

Para cada uno, seleccionen el/los hiperparámetro(s) óptimo(s) usando *cross-validation* sobre `train` (`cv=5` es un buen punto de partida):

1. **Regresión Lineal** (sin regularizar) — línea base.
2. **Ridge** — barrer un rango de `alpha` (ej. `np.logspace(-3, 3, 50)`).
3. **Lasso** — mismo rango de `alpha`.
4. **Elastic Net** — barrer `alpha` y `l1_ratio` (ej. `[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]`).
5. **KNN Regressor** — barrer `n_neighbors` (ej. de 1 a 30) usando `GridSearchCV` o `cross_val_score` dentro de un pipeline `preprocessor + KNeighborsRegressor`.

> Pista: `KNeighborsRegressor` **depende fuertemente del escalamiento** — sin `StandardScaler` en el pipeline sus resultados serían muy distintos. Piensen por qué.

### 5. Evaluación

Para cada modelo, calcular sobre el conjunto de **prueba**:

- **MAE** (`mean_absolute_error`)
- **R²** (`r2_score`)

Además, para evaluar **estabilidad** (no solo el desempeño puntual en test), calculen con `cross_val_score` sobre `train` la media y la desviación estándar de MAE y R² a través de los folds de cada modelo.

Construyan una tabla resumen con columnas: `Modelo | Hiperparámetro(s) elegido(s) | MAE (test) | R² (test) | MAE CV (media ± std) | R² CV (media ± std)`.

### 6. Comparación de coeficientes vs. vecinos

- Para los modelos lineales, extraer y graficar el vector de pesos (`coef_`) de cada uno, igual que en el notebook de ejemplo.
- El KNN no tiene coeficientes — en su lugar, discutan (en el análisis, no en código) cómo KNN "usa" las variables: todas contribuyen a la distancia euclidiana con el mismo peso relativo después del escalamiento, no hay selección de variables como en Lasso.

### 7. ¿Cuál modelo es más robusto?

Esta es la pregunta central del ejercicio. **Robusto** aquí significa: buen desempeño en test, consistente con el desempeño en CV (poca diferencia entre ambos, y poca varianza entre folds), y que no depende de forma crítica de acertar exactamente el hiperparámetro óptimo (un modelo cuyo error se dispara si `alpha` o `k` cambian ligeramente es frágil, aunque su punto óptimo se vea bien).

Para sustentar su respuesta:

- Comparen la tabla de la sección 5: ¿qué modelo tiene menor `std` en CV?
- Grafiquen MAE o R² de CV en función del hiperparámetro (curva de `alpha` para Ridge/Lasso, curva de `k` para KNN) — ¿qué tan plana o sensible es cada curva cerca del óptimo?
- Consideren el tamaño del dataset (filas) y el número de variables: ¿favorece esto a un modelo paramétrico (regularización) o a uno no paramétrico (KNN)?
- ¿Alguno de los modelos sería más sensible a datos nuevos fuera del rango visto en entrenamiento (extrapolación)? KNN vs. modelos lineales se comportan muy distinto aquí.

## Entregable

Un notebook (`.ipynb`) con:

1. Código completo y ejecutado (con outputs) siguiendo los pasos 1–6.
2. La tabla comparativa de la sección 5.
3. Los gráficos de coeficientes y de sensibilidad al hiperparámetro.
4. Una celda de markdown final (mínimo 150 palabras) respondiendo explícitamente: **¿cuál modelo es el más robusto y por qué?**, usando evidencia de las tablas y gráficos generados — no basta con decir "el que tiene mejor R² en test".

## Criterios de evaluación

| Criterio | Peso |
|---|---|
| Pipeline de preprocesamiento correcto (sin fuga de datos entre train/test) | 20% |
| Selección de hiperparámetros vía CV para los 5 modelos | 25% |
| Tabla comparativa completa (MAE, R², test y CV) | 20% |
| Gráficos de coeficientes y de sensibilidad al hiperparámetro | 15% |
| Calidad del argumento de robustez (usa evidencia, no solo opinión) | 20% |
