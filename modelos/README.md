# Modelos (Python)

Cada notebook carga el split, ajusta sus modelos, los evalúa en la validación y guarda **sólo las predicciones**, partido a partido. Las métricas salen siempre de la misma función. En todos los modelos la clase predicha sale de un umbral de 0,5, y los modelos se comparan por log-loss. El test se usa una sola vez, para el modelo elegido.

El código está en `codigo/` y los HTML exportados quedan en `informes/`. Cómo correr los notebooks está en el [README principal](../README.md#correr-todo).

## tesina.py · Lo común

Las funciones que usan todos los notebooks:

- `cargar_split()` lee la matriz y devuelve train, validación y test.
- `evaluar()` calcula las métricas: log-loss, Brier, AUC, accuracy y compañía.
- `guardar_predicciones()` guarda las predicciones de un modelo, partido a partido.
- `evaluar_corrida()` evalúa todas las predicciones guardadas.
- `umbral_optimo()` da el umbral que maximizaría el accuracy en la validación. Se muestra sólo como señal de calibración: en un modelo calibrado da cerca de 0,5.

## 01_modelos_base.py · Modelos base

Las reglas a superar. Primero las que no miran el partido: siempre gana el local (MB0) y gana el mejor posicionado en la temporada anterior (MB1). Después las que usan el corte: gana el que va ganando (MB2) y gana el de mejor puntaje de los cuatro factores de Oliver (MB3).

## 02_logistica.py · Regresión logística

Una sola regresión logística. Sus variables se eligen por el método de Collett, con tests de razón de verosimilitud, entre 9 candidatas definidas de antemano y sin pares muy correlacionados: el marcador al corte, el contexto (playoffs y el promedio de partidos jugados de los dos equipos), pérdidas, rebote ofensivo y libres, y la fuerza previa por margen medio. Las pérdidas, el rebote, los libres y la fuerza entran como diferencia local − visita.

## 03_flaml.py · Árboles con FLAML

FLAML elige, entre XGBoost, LightGBM y bosque aleatorio, la familia y los hiperparámetros que minimizan el log-loss, con todas las variables numéricas de la matriz. Para tunear valida con origen móvil expandido dentro del train, no con k-fold, que con partidos ordenados en el tiempo mete el futuro en el entrenamiento.

## resultados.py · Informe

Lee todas las predicciones y arma las tablas de métricas, ordenadas por log-loss: primero los modelos que no miran el partido (MB0 y MB1) y después el resto.

## exploracion_tiro.py · Exploración

Cómo se relaciona la diferencia al 2° cuarto con el resultado y con las estadísticas de tiro, sólo sobre el train. No guarda predicciones; se exporta igual que los demás notebooks.

`02_xgboost.py` y `03_tabpfn.py` son versiones viejas que ya no andan. El primero ya lo reemplaza `03_flaml.py`; el segundo se va a reemplazar por un notebook nuevo. TabPFN se instala aparte, con `uv sync --extra tabpfn`, porque trae torch.

## Archivos generados

| Archivo | Descripción |
| --- | --- |
| `predicciones/2c/<conjunto>/pred_<modelo>.csv` | Predicciones de cada modelo, partido a partido (`<conjunto>` es `validacion` o `test`) |
| `predicciones/2c/<conjunto>/pred_<modelo>.notas.txt` | Qué variables usa cada modelo |
| `modelos/informes/*.html` | Los notebooks exportados, con código y salidas |
