# Ejercicio: Clustering con Iris — Agglomerative Clustering vs. DBSCAN

## Objetivo

Aplicar y comparar dos estrategias de **clustering no supervisado** sobre el dataset `Iris`:

- **Clustering jerárquico aglomerativo**
- **DBSCAN**

Al finalizar el ejercicio, deberán analizar los resultados obtenidos y justificar qué método consideran más apropiado para este conjunto de datos.

---

## 1. Dataset

Utilicen el dataset **Iris**, disponible directamente en `scikit-learn`.

El dataset contiene mediciones de flores correspondientes a cuatro características:

- `sepal length`
- `sepal width`
- `petal length`
- `petal width`

El dataset contiene 150 observaciones.

> **Importante:** para realizar el clustering deben utilizar únicamente las características (`X`). Las etiquetas reales (`y`) **no deben utilizarse durante el entrenamiento del clustering**.

Pueden cargar el dataset con:

```python
from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
y = iris.target
```

Aunque `y` está disponible, se utilizará únicamente como referencia opcional al final del ejercicio, no para construir los clústeres.

---

# 2. Preparación de los datos

Antes de aplicar los algoritmos:

1. Explore las dimensiones del dataset.
2. Revise las variables disponibles.
3. Verifique si existen valores faltantes.
4. Estandarice las características utilizando `StandardScaler`.

Por ejemplo:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Pregunta

¿Por qué es importante estandarizar las variables antes de aplicar métodos basados en distancias?

---

# 3. Clustering jerárquico aglomerativo

Utilice `AgglomerativeClustering` de `scikit-learn`.

Inicialmente deben explorar diferentes cantidades de clústeres.

Prueben, como mínimo:

```text
k = 2, 3, 4, 5, 6
```

Utilicen:

```python
from sklearn.cluster import AgglomerativeClustering

model = AgglomerativeClustering(
    n_clusters=k,
    linkage="ward"
)

labels = model.fit_predict(X_scaled)
```

## 3.1. Dendrograma

Construyan un **dendrograma** utilizando `scipy`.

```python
from scipy.cluster.hierarchy import dendrogram, linkage

Z = linkage(X_scaled, method="ward")

dendrogram(Z)
```

### Analicen

- ¿Cuántos grupos parecen existir visualmente?
- ¿En qué alturas del dendrograma se producen las principales fusiones?
- ¿Qué número de clústeres sugerirían inicialmente a partir del dendrograma?

---

# 4. Evaluación con Silhouette Score

Para cada valor de `k`, calculen el **Silhouette Score**.

```python
from sklearn.metrics import silhouette_score

score = silhouette_score(X_scaled, labels)
```

Construyan una tabla similar a:

| Número de clusters | Silhouette Score |
|---:|---:|
| 2 | ... |
| 3 | ... |
| 4 | ... |
| 5 | ... |
| 6 | ... |

También pueden realizar una gráfica:

```text
Número de clusters vs. Silhouette Score
```

### Preguntas

1. ¿Qué valor de `k` obtiene el mayor Silhouette Score?
2. ¿Coincide con la interpretación realizada a partir del dendrograma?
3. ¿Qué número de clústeres seleccionarían finalmente para Agglomerative Clustering?
4. Justifiquen la selección.

---

# 5. Visualización de los clústeres

Para el número de clústeres seleccionado, grafiquen los resultados.

Como Iris tiene cuatro características, pueden utilizar, por ejemplo:

- `petal length` vs. `petal width`

o utilizar una técnica de reducción dimensional como **PCA** para representar las cuatro características en dos dimensiones.

La gráfica debe permitir identificar visualmente los grupos encontrados.

Incluyan:

- Los puntos de las observaciones.
- Un color diferente para cada clúster.
- Una leyenda.
- Título y nombres de los ejes.

---

# 6. DBSCAN

Ahora repitan el análisis utilizando **DBSCAN**.

Importen:

```python
from sklearn.cluster import DBSCAN
```

DBSCAN depende principalmente de dos parámetros:

- `eps`
- `min_samples`

Realicen una exploración de diferentes combinaciones.

Por ejemplo:

```text
eps = 0.2, 0.3, 0.4, 0.5, 0.6
min_samples = 3, 5, 8
```

No es necesario probar todas las combinaciones posibles si encuentran rápidamente una región de parámetros razonable.

Ejemplo:

```python
model = DBSCAN(
    eps=0.5,
    min_samples=5
)

labels = model.fit_predict(X_scaled)
```

---

# 7. Analizar el resultado de DBSCAN

Recuerden que DBSCAN puede producir puntos etiquetados como:

```text
-1
```

Estos corresponden a **ruido/outliers**.

Por lo tanto, deben determinar:

1. ¿Cuántos clústeres encontró DBSCAN?
2. ¿Cuántos puntos fueron considerados ruido?
3. ¿Qué combinación de `eps` y `min_samples` produjo el resultado seleccionado?
4. ¿Cómo cambia el número de clústeres cuando modifican `eps`?
5. ¿Cómo cambia el resultado cuando modifican `min_samples`?

---

# 8. Silhouette Score para DBSCAN

Calculen el Silhouette Score para los resultados de DBSCAN.

Tengan en cuenta que el valor `-1` representa ruido.

Para una comparación más informativa, pueden calcular el Silhouette Score sobre los puntos que **no fueron clasificados como ruido**:

```python
mask = labels != -1

score = silhouette_score(
    X_scaled[mask],
    labels[mask]
)
```

### Importante

Si DBSCAN produce solamente un clúster válido, o ningún clúster válido, el Silhouette Score no puede calcularse de forma convencional.

Su código debe contemplar este caso.

---

# 9. Comparación de los dos métodos

Construyan una tabla final que permita comparar ambos métodos.

Por ejemplo:

| Método | Parámetros | Nº clusters | Nº puntos ruido | Silhouette Score |
|---|---|---:|---:|---:|
| Agglomerative | `k = ...` | ... | N/A | ... |
| DBSCAN | `eps = ..., min_samples = ...` | ... | ... | ... |

Además, incluyan una gráfica para cada método utilizando el mismo espacio de características para facilitar la comparación.

---

# 10. Análisis de resultados

Respondan las siguientes preguntas.

### A. Agglomerative Clustering

1. ¿Cuál fue el número de clústeres seleccionado?
2. ¿Por qué?
3. ¿Qué información aportó el dendrograma?
4. ¿Qué información aportó el Silhouette Score?

### B. DBSCAN

1. ¿Qué valores de `eps` y `min_samples` seleccionaron?
2. ¿Cuántos clústeres encontraron?
3. ¿DBSCAN identificó puntos como ruido?
4. ¿Qué efecto tuvo modificar `eps`?
5. ¿Qué efecto tuvo modificar `min_samples`?

### C. Comparación

Comparen ambos métodos considerando:

- Número de clústeres encontrados.
- Silhouette Score.
- Cantidad de ruido detectado.
- Forma de los grupos.
- Dependencia de parámetros.
- Interpretación de los resultados.

### Pregunta principal

> **Si tuvieran que utilizar uno de los dos métodos para analizar este dataset, ¿cuál preferirían y por qué?**

La respuesta debe estar **justificada con los resultados obtenidos**, no únicamente con una descripción teórica de los algoritmos.

---

# 11. Datos de prueba

Finalmente, inventen **al menos tres nuevas flores** que no pertenezcan al dataset original.

Cada flor debe tener las cuatro características:

```text
sepal length
sepal width
petal length
petal width
```

Por ejemplo:

```python
X_test = [
    [5.0, 3.4, 1.5, 0.2],
    [6.0, 2.8, 4.5, 1.5],
    [6.7, 3.0, 5.8, 2.0]
]
```

**No es obligatorio utilizar estos valores.** Pueden crear sus propios datos.

---

# 12. Predicción del clúster para los nuevos datos

Transformen los nuevos datos utilizando **el mismo `StandardScaler` utilizado para entrenar el clustering**:

```python
X_test_scaled = scaler.transform(X_test)
```

### Agglomerative Clustering

Analicen cómo pueden asignar cada nuevo punto a uno de los clústeres encontrados.

> **Importante:** `AgglomerativeClustering` de `scikit-learn` no proporciona un método `.predict()` convencional para nuevas observaciones. Por lo tanto, deberán proponer e implementar una estrategia razonable para asignar los nuevos puntos a los clústeres obtenidos.

Expliquen brevemente la estrategia utilizada.

---

### DBSCAN

Analicen también cómo clasificarían los nuevos puntos utilizando el resultado obtenido con DBSCAN.

> **Importante:** `DBSCAN` de `scikit-learn` tampoco proporciona un `.predict()` convencional para nuevas observaciones. Por lo tanto, deberán investigar y justificar una estrategia para asignar nuevas observaciones a los grupos encontrados.


---

## Pregunta final de reflexión

> **¿El método que obtuvo el mayor Silhouette Score es necesariamente el método que ustedes preferirían utilizar?**

Justifiquen su respuesta considerando no solamente la métrica, sino también las características del problema y el comportamiento observado en las visualizaciones.

