# Tesina ACB · ¿Quién gana, visto desde el medio tiempo?

Este repo tiene el código de mi tesina de la Licenciatura en Estadística (UNR), dirigida por Tomás Capretto. La pregunta es simple: **con lo que se sabe al terminar el 2° cuarto de un partido de la Liga Endesa (ACB), ¿qué tan bien se puede predecir quién gana?**, y cuánto mejora eso respecto de reglas simples como "gana el que va ganando". Más adelante, la idea es cruzar esas probabilidades con las cuotas de las casas de apuestas.

Todo se reproduce desde cero: el repo no guarda datos ni resultados, sólo código. **R** baja el play-by-play de la API oficial de la ACB y arma los datos, y **Python** ajusta y evalúa los modelos en notebooks de [marimo](https://marimo.io). El scraping parte de [openacb](https://github.com/juantorrecillas/openacb), de Juan Torrecillas.

## Cómo hacerlo andar

Los comandos de esta sección son iguales en PowerShell, en cmd y en bash o zsh, salvo donde se indica otra cosa, y se corren todos **desde la carpeta del repo**.

### Requisitos

- **[R](https://cran.r-project.org)**.
- **[uv](https://docs.astral.sh/uv/)**, que instala Python y todas sus dependencias: no hace falta tener Python instalado.
- **Git**, para clonar el repo.
- **`ACB_TOKEN`**: el token bearer de la API de la ACB (`api2.acb.com`). Sólo hace falta para bajar los datos.

### Instalación (una sola vez)

1. Clonar el repo y entrar en su carpeta:

   ```
   git clone https://github.com/franovando/liga_acb_prediction.git
   cd liga_acb_prediction
   ```

2. Copiar `.Renviron.example` a `.Renviron` y completar ahí `ACB_TOKEN`, sin el prefijo `Bearer`. En PowerShell, bash o zsh:

   ```
   cp .Renviron.example .Renviron
   ```

   En cmd:

   ```
   copy .Renviron.example .Renviron
   ```

   `.Renviron` no se sube al repo: está en el `.gitignore`.

3. Instalar los paquetes de R. Esto se corre en una consola de R (o en RStudio), no en la terminal:

   ```r
   install.packages(c("dplyr", "tidyr", "tibble", "httr", "jsonlite", "foreach", "doParallel"))
   ```

4. Armar el entorno de Python:

   ```
   uv sync
   ```

### Correr todo

Primero los datos. Este comando baja las cinco temporadas (unos 1.600 partidos, en paralelo) y arma todo `datos/`:

```
Rscript pipeline.R
```

Después los notebooks, en este orden. FLAML tarda varios minutos, y `resultados.py` va al final porque lee las predicciones que dejan los demás:

```
uv run marimo export html modelos/codigo/01_modelos_base.py -o modelos/informes/01_modelos_base.html -f
uv run marimo export html modelos/codigo/02_logistica.py -o modelos/informes/02_logistica.html -f
uv run marimo export html modelos/codigo/03_flaml.py -o modelos/informes/03_flaml.html -f
uv run marimo export html modelos/codigo/resultados.py -o modelos/informes/resultados.html -f
```

Al final, `modelos/informes/resultados.html` tiene la tabla con todos los modelos. Los HTML necesitan internet para verse, porque la interfaz de marimo se baja al abrirlos.

Algunas cosas a tener en cuenta:

- R lee `.Renviron` de la carpeta en la que arranca. Si se corre `pipeline.R` desde otra carpeta, `ACB_TOKEN` queda vacío y el pipeline corta con un aviso.
- Si la terminal no encuentra `Rscript` (suele pasar en Windows), hay que agregar la carpeta `bin` de R al `PATH` o llamarlo con la ruta completa. En PowerShell, por ejemplo, `& "C:\Program Files\R\R-4.5.1\bin\Rscript.exe" pipeline.R`; en cmd, lo mismo sin el `&` del principio. La versión de R en la ruta depende de la instalación.
- Una vez bajados los datos, **`Rscript pipeline.R --no-scraping`** rearma todo lo demás en segundos, sin tocar la API. Es lo que hay que correr cada vez que cambia algo en `etl/`.
- Para abrir un notebook y explorarlo en vez de exportarlo: `uv run marimo edit modelos/codigo/02_logistica.py`.

## Estructura del proyecto

```
liga_acb_prediction/
├── pipeline.R                  # Arma todos los datos: Rscript pipeline.R [--no-scraping]
├── etl/                        # R: de la API de la ACB a la matriz de features, en orden
│   ├── config/seasons.R
│   ├── 01_scrape.R
│   ├── 03_clean.R
│   ├── 04_box_score.R
│   ├── 05_datos.R
│   └── 06_split.R
├── modelos/
│   ├── codigo/                 # Python: funciones comunes y notebooks de marimo
│   │   ├── tesina.py
│   │   ├── 01_modelos_base.py
│   │   ├── 02_logistica.py
│   │   ├── 03_flaml.py
│   │   ├── resultados.py
│   │   ├── exploracion_tiro.py
│   │   ├── 02_xgboost.py       # Versión vieja, ya no anda
│   │   └── 03_tabpfn.py        # Versión vieja, ya no anda
│   └── informes/               # HTML exportados de los notebooks (no se versionan)
├── cuotas/                     # Python: captura de cuotas en vivo
│   ├── capturar.py
│   └── programar.ps1
├── datos/                      # Lo arma pipeline.R (no se versiona)
├── predicciones/               # Lo escriben los notebooks (no se versiona)
├── .Renviron.example           # Plantilla para el token de la API de la ACB
├── pyproject.toml              # Dependencias de Python
└── uv.lock                     # Versiones exactas de esas dependencias
```

## Pipeline de datos (R)

`pipeline.R` corre los pasos en orden, sobre las temporadas 2021-22 a 2025-26, con el corte al final del 2° cuarto.

### 01_scrape.R · Descarga

Baja el play-by-play de cada partido terminado desde `api2.acb.com`, en paralelo y con reintentos; si un partido falla igual, corta. Guarda un CSV por partido en una carpeta temporal.

### 03_clean.R · Limpieza

Junta los partidos de cada temporada y crea las columnas de eventos de box score (tiros, rebotes, pérdidas, asistencias, etc.). Deja un play-by-play limpio por temporada en `datos/pbp/`.

### 04_box_score.R · Box score al corte

Pasa del play-by-play a una fila por partido, con todo acumulado **hasta el final del cuarto de corte**, más la diferencia de puntos al cierre de cada cuarto. El marcador de cada cuarto se toma del último evento según el orden de la jugada, porque la ACB a veces carga marcadores erróneos en el medio.

### 05_datos.R · Matriz de features

Arma las variables que ven los modelos:

- la diferencia y el total de puntos al corte, y la diferencia al final del 1er cuarto;
- el box score y los cuatro factores de Dean Oliver de cada equipo;
- la fase del torneo, los partidos jugados y el porcentaje de victorias de la temporada anterior;
- la fuerza previa de cada equipo: el margen medio y la proporción de victorias de la temporada hasta el partido anterior, arrancando desde la temporada anterior.

También unifica los nombres de los equipos, que cambian con cada sponsor.

### 06_split.R · Split

Divide por temporadas enteras: **train** 2021-22 a 2023-24, **validación** 2024-25 y **test** 2025-26. Exporta la matriz para Python, con el conjunto de cada partido marcado.

## Modelos (Python)

Cada notebook carga el split, ajusta sus modelos, los evalúa en la validación y guarda **sólo las predicciones**, partido a partido. Las métricas salen siempre de la misma función. En todos los modelos la clase predicha sale de un umbral de 0,5, y los modelos se comparan por log-loss. El test se usa una sola vez, para el modelo elegido.

### tesina.py · Lo común

Las funciones que usan todos los notebooks:

- `cargar_split()` lee la matriz y devuelve train, validación y test.
- `evaluar()` calcula las métricas: log-loss, Brier, AUC, accuracy y compañía.
- `guardar_predicciones()` guarda las predicciones de un modelo, partido a partido.
- `evaluar_corrida()` evalúa todas las predicciones guardadas.
- `umbral_optimo()` da el umbral que maximizaría el accuracy en la validación. Se muestra sólo como señal de calibración: en un modelo calibrado da cerca de 0,5.

### 01_modelos_base.py · Modelos base

Las reglas a superar. Primero las que no miran el partido: siempre gana el local (MB0) y gana el mejor posicionado en la temporada anterior (MB1). Después las que usan el corte: gana el que va ganando (MB2) y gana el de mejor puntaje de los cuatro factores de Oliver (MB3).

### 02_logistica.py · Regresión logística

Una sola regresión logística. Sus variables se eligen por el método de Collett, con tests de razón de verosimilitud, entre 9 candidatas definidas de antemano y sin pares muy correlacionados: el marcador al corte, el contexto (playoffs y el promedio de partidos jugados de los dos equipos), pérdidas, rebote ofensivo y libres, y la fuerza previa por margen medio. Las pérdidas, el rebote, los libres y la fuerza entran como diferencia local − visita.

### 03_flaml.py · Árboles con FLAML

FLAML elige, entre XGBoost, LightGBM y bosque aleatorio, la familia y los hiperparámetros que minimizan el log-loss, con todas las variables numéricas de la matriz. Para tunear valida con origen móvil expandido dentro del train, no con k-fold, que con partidos ordenados en el tiempo mete el futuro en el entrenamiento.

### resultados.py · Informe

Lee todas las predicciones y arma las tablas de métricas, ordenadas por log-loss: primero los modelos que no miran el partido (MB0 y MB1) y después el resto.

### exploracion_tiro.py · Exploración

Cómo se relaciona la diferencia al 2° cuarto con el resultado y con las estadísticas de tiro, sólo sobre el train. No guarda predicciones; se exporta igual que los demás notebooks.

`02_xgboost.py` y `03_tabpfn.py` son versiones viejas que ya no andan. El primero ya lo reemplaza `03_flaml.py`; el segundo se va a reemplazar por un notebook nuevo. TabPFN se instala aparte, con `uv sync --extra tabpfn`, porque trae torch.

## Cuotas en vivo (Python)

Las cuotas en los cortes no se consiguen después del partido, así que se capturan en el momento. Esto corre aparte del pipeline, en una notebook con Windows y sin intervención: una tarea programada lanza el script cada mañana, y el script despierta la notebook para cada captura y la deja suspendida el resto del tiempo.

Está hecho y probado en Windows, con Python (vía uv), Playwright con Chromium para leer Jugadón, el Programador de tareas de Windows (lo configura `programar.ps1`) y los temporizadores de reactivación de Windows para despertar la notebook. Los comandos de esta sección son para PowerShell.

### capturar.py · BetWarrior y Jugadón

Guarda las cuotas de ganador, hándicap y total (con prórroga incluida y todas sus líneas) de BetWarrior y Jugadón, en cuatro momentos de cada partido de la ACB:

- `inicio`: diez minutos antes de la hora, porque hay partidos que empiezan antes;
- `fin_2C`: al terminar el 2° cuarto;
- `fin_HT`: 13 minutos después, poco antes del 3er cuarto;
- `fin_3C`: en la pausa entre el 3er y el 4to cuarto.

El reloj sale de BetWarrior, que corre sobre Kambi y publica su oferta en JSON sin login. Jugadón sólo responde a los pedidos que salen de su página, así que el script la abre en un navegador sin ventana (Chromium, con Playwright). Si el mercado de ganador está suspendido, reintenta cada 30 segundos. Corre una sola captura a la vez, captura sólo los partidos del día y termina.

Entre una captura y la siguiente, deja que la notebook se suspenda y la despierta con un temporizador de Windows. El inicio tiene hora fija; para los cortes usa el reloj del partido, sabiendo que el reloj de juego nunca corre más rápido que el real. Nunca fuerza la suspensión: si la notebook se está usando, sigue prendida.

Antes de la primera captura hay que bajar el navegador:

```powershell
uv run playwright install chromium
```

Para correrlo a mano, con la Liga ACB:

```powershell
uv run python cuotas/capturar.py
```

Para probarlo con otra liga se le pasan dos argumentos, la liga en Kambi y el número del torneo en Jugadón (el de la Liga ACB es 3111). Por ejemplo, para la Euroliga: `uv run python cuotas/capturar.py basketball/euroleague/all/all ID_JUGADON`, con el número del torneo en lugar de `ID_JUGADON`.

### programar.ps1 · La tarea de Windows

Crea la tarea `tesina-cuotas`, que todos los días a las 6:30 despierta la notebook si está suspendida y corre `capturar.py`; si ese día no hay partidos, termina enseguida y Windows la vuelve a suspender. También corre al iniciar sesión, para retomar el día si la notebook se apagó en el medio (en ese caso puede repetirse una captura ya hecha).

La notebook tiene que quedar suspendida (no apagada, porque apagada Windows no la puede despertar), enchufada y con los temporizadores de reactivación permitidos. Todo esto se corre una sola vez, en una PowerShell abierta como administrador, desde la raíz del repo:

```powershell
# El botón de encendido y la tapa suspenden, enchufada y con batería
powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS PBUTTONACTION 1
powercfg /setdcvalueindex SCHEME_CURRENT SUB_BUTTONS PBUTTONACTION 1
powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 1
powercfg /setdcvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 1

# Los temporizadores pueden despertar la notebook. Con batería, gasta batería: cuidado en la mochila
powercfg /setacvalueindex SCHEME_CURRENT SUB_SLEEP RTCWAKE 1
powercfg /setdcvalueindex SCHEME_CURRENT SUB_SLEEP RTCWAKE 1
powercfg /setactive SCHEME_CURRENT

# La tarea
powershell -ExecutionPolicy Bypass -File cuotas\programar.ps1
```

## Archivos generados

| Archivo | Descripción |
| --- | --- |
| `datos/pbp/PBP_Data_<año>.csv` | Play-by-play limpio, uno por temporada, con el año en que termina (2022 es la 2021-22); unos 110 MB cada uno |
| `datos/pbp/PBP_Data_21-26.rds` | Las cinco temporadas juntas: caché de R para no releer los CSV |
| `datos/box_scores/box_score_2c_21-26.csv` | Una fila por partido, con todo acumulado hasta el corte |
| `datos/matrices/matriz_2c.csv` | Matriz de features con el split marcado: lo que leen los notebooks |
| `predicciones/2c/<conjunto>/pred_<modelo>.csv` | Predicciones de cada modelo, partido a partido (`<conjunto>` es `validacion` o `test`) |
| `predicciones/2c/<conjunto>/pred_<modelo>.notas.txt` | Qué variables usa cada modelo |
| `modelos/informes/*.html` | Los notebooks exportados, con código y salidas |
| `datos/cuotas/betwarrior.csv` y `datos/cuotas/jugadon.csv` | Cuotas de cada casa: una fila por partido, captura, mercado y línea |
| `datos/cuotas/capturar.log` | Lo que imprime `capturar.py` cuando lo lanza la tarea de Windows |

*hecho por Fran Ovando 🧉*
