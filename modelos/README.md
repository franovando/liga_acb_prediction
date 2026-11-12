# Modelos (Python)

Cada notebook carga el split, ajusta sus modelos, los evalúa en la validación y guarda **sólo las predicciones**, partido a partido. Las métricas salen siempre de la misma función. En todos los modelos la clase predicha sale de un umbral de 0,5, y los modelos se comparan por log-loss. El test se usa una sola vez, para el modelo elegido.

El código está en `codigo/` y los HTML exportados quedan en `informes/`. Cómo correr los notebooks está en el [README principal](../README.md#correr-todo).

## tesina.py · Lo común

Las funciones que usan todos los notebooks:

- `cargar_split()` lee la matriz y devuelve train, validación y test.
- `evaluar()` calcula las métricas: log-loss, Brier, AUC, accuracy y compañía.
- `guardar_predicciones()` guarda las predicciones de un modelo, partido a partido.
- `evaluar_corrida()` evalúa todas las predicciones guardadas.
- `guardar_predictiva()` guarda, para los modelos de la diferencia de puntos final que dan una distribución (los bayesianos), simulaciones de la predictiva y la log verosimilitud de lo observado, partido a partido.
- `evaluar_predictiva()` evalúa esas predictivas: log densidad predictiva, cobertura de los intervalos del 50 % y del 90 %, y PIT.
- `umbral_optimo()` da el umbral que maximizaría el accuracy en la validación. Se muestra sólo como señal de calibración: en un modelo calibrado da cerca de 0,5.

## 01_modelos_base.py · Modelos base

Las reglas a superar. Primero las que no miran el partido: siempre gana el local (MB0) y gana el mejor posicionado, el de mayor proporción de victorias en sus últimos 34 partidos de fase regular (MB1). Después las que usan el corte: gana el que va ganando (MB2), gana el de mejor puntaje de los cuatro factores de Oliver (MB3) y la votación de MB0, MB1 y MB2 (MB4).

## 02_logistica.py · Regresión logística

Una sola regresión logística, con 9 variables definidas de antemano, en términos del problema, y sin selección: elegir las que dan significativas sobre los mismos datos sesga sus coeficientes. No hay pares muy correlacionados: el marcador al corte y quién ganó el 1er cuarto, el contexto (playoffs y el promedio de partidos jugados de los dos equipos), pérdidas, rebote ofensivo y libres, y la fuerza previa por margen medio. Las pérdidas, el rebote, los libres y la fuerza entran como diferencia local − visita.

## 03_flaml.py · Árboles con FLAML

FLAML elige, entre XGBoost, LightGBM y bosque aleatorio, la familia y los hiperparámetros que minimizan el log-loss, con todas las variables numéricas de la matriz. Para tunear valida con origen móvil expandido dentro del train, no con k-fold, que con partidos ordenados en el tiempo mete el futuro en el entrenamiento.

## 04_bayesiano.py · Diferencia de puntos final

Una regresión bayesiana, en PyMC, de la diferencia final local − visita, con la prórroga incluida, con error t de Student y las mismas variables que la logística (`t_lineal`). Las variables continuas se estandarizan para muestrear y los coeficientes se informan en sus unidades originales. Tiene la predictiva a priori, el diagnóstico de convergencia y chequeos predictivos a posteriori sobre el train. De la distribución predictiva sale la probabilidad de victoria, P(diferencia final > 0), que se guarda como la de cualquier modelo; la predictiva entera se guarda aparte con `guardar_predictiva()`.

## resultados.py · Informe

Lee todas las predicciones y arma las tablas de métricas, ordenadas por log-loss: primero los modelos que no miran el partido (MB0 y MB1) y después el resto. Al final, el gráfico de calibración de los modelos que dan probabilidades: los partidos de la validación en 10 grupos de igual cantidad según la probabilidad predicha, y en cada grupo la proporción que ganó el local, contra la probabilidad media. Y, para los modelos de la diferencia de puntos final, la evaluación de su distribución predictiva, con el histograma del PIT.

## exploracion_tiro.py · Exploración

Cómo se relaciona la diferencia al 2° cuarto con el resultado y con las estadísticas de tiro, sólo sobre el train. No guarda predicciones; se exporta igual que los demás notebooks.

`02_xgboost.py` es una versión vieja que ya no anda; lo reemplaza `03_flaml.py`.

## Archivos generados

| Archivo | Descripción |
| --- | --- |
| `predicciones/2c/<conjunto>/pred_<modelo>.csv` | Predicciones de cada modelo, partido a partido (`<conjunto>` es `validacion` o `test`) |
| `predicciones/2c/<conjunto>/pred_<modelo>.notas.txt` | Qué variables usa cada modelo |
| `predicciones/2c/<conjunto>/sim_<modelo>.csv` | La predictiva de los modelos de la diferencia de puntos final: 2.000 simulaciones por partido y la log verosimilitud de lo observado |
| `modelos/informes/*.html` | Los notebooks exportados, con código y salidas |
