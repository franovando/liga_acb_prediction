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

Qué hace cada archivo está en el README de su carpeta:

- [`etl/`](etl/README.md): el pipeline de datos en R, del play-by-play al split.
- [`modelos/`](modelos/README.md): los notebooks de los modelos y el informe de resultados.
- [`cuotas/`](cuotas/README.md): la captura de cuotas en vivo.

*hecho por Fran Ovando 🧉*
