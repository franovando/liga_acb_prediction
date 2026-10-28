# Pipeline de datos (R)

`pipeline.R`, en la raíz del repo, corre estos pasos en orden, sobre las temporadas 2021-22 a 2025-26, con el corte al final del 2° cuarto. Cómo correrlo está en el [README principal](../README.md#correr-todo).

## 01_scrape.R · Descarga

Baja el play-by-play de cada partido terminado desde `api2.acb.com`, en paralelo y con reintentos; si un partido falla igual, corta. Guarda un CSV por partido en una carpeta temporal.

## 03_clean.R · Limpieza

Junta los partidos de cada temporada y crea las columnas de eventos de box score (tiros, rebotes, pérdidas, asistencias, etc.). Deja un play-by-play limpio por temporada en `datos/pbp/`.

## 04_box_score.R · Box score al corte

Pasa del play-by-play a una fila por partido, con todo acumulado **hasta el final del cuarto de corte**, más la diferencia de puntos al cierre de cada cuarto. El marcador de cada cuarto se toma del último evento según el orden de la jugada, porque la ACB a veces carga marcadores erróneos en el medio.

## 05_datos.R · Matriz de features

Arma las variables que ven los modelos:

- la diferencia y el total de puntos al corte, y la diferencia al final del 1er cuarto;
- el box score y los cuatro factores de Dean Oliver de cada equipo;
- la fase del torneo, los partidos jugados y el porcentaje de victorias de la temporada anterior;
- la fuerza previa de cada equipo: el margen medio y la proporción de victorias de la temporada hasta el partido anterior, arrancando desde la temporada anterior.

También unifica los nombres de los equipos, que cambian con cada sponsor.

## 06_split.R · Split

Divide por temporadas enteras: **train** 2021-22 a 2023-24, **validación** 2024-25 y **test** 2025-26. Exporta la matriz para Python, con el conjunto de cada partido marcado.

## Archivos generados

| Archivo | Descripción |
| --- | --- |
| `datos/pbp/PBP_Data_<año>.csv` | Play-by-play limpio, uno por temporada, con el año en que termina (2022 es la 2021-22); unos 110 MB cada uno |
| `datos/pbp/PBP_Data_21-26.rds` | Las cinco temporadas juntas: caché de R para no releer los CSV |
| `datos/box_scores/box_score_2c_21-26.csv` | Una fila por partido, con todo acumulado hasta el corte |
| `datos/matrices/matriz_2c.csv` | Matriz de features con el split marcado: lo que leen los notebooks |
