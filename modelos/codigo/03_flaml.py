import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell
def _():
    import warnings

    import marimo as mo
    import numpy as np
    import polars as pl
    from flaml import AutoML

    import tesina

    # FLAML con scikit-learn 1.9 avisa esto miles de veces por corrida, y el aviso
    # entra entero en el HTML (25 MB). No cambia ningún resultado.
    warnings.filterwarnings("ignore", message=r".*sklearn\.utils\.parallel\.delayed.*")

    return AutoML, mo, np, pl, tesina


@app.cell
def _(mo):
    mo.md("""
    # FLAML: árboles con tuneo automático

    FLAML busca, entre tres familias de árboles (XGBoost, LightGBM y bosque
    aleatorio), la familia y los hiperparámetros que minimizan el **log-loss**.
    Se ajusta sobre el train y se evalúa en la **validación** (la temporada
    2024-25 entera), igual que el resto de los modelos. El **test** no se usa
    acá.

    - **Variables:** todas las numéricas de la matriz, menos los equipos (por
      ahora no se usan como variables): el marcador al corte, la diferencia
      al final del 1er cuarto, el box score, los cuatro factores, `es_playoff`,
      `partidos_jugados`, `pct_victorias_anterior` (que es `NaN` para los
      recién ascendidos: los árboles lo manejan) y la fuerza previa de cada
      equipo (`fuerza_margen` y `fuerza_victorias`).
    - **Validación interna: rolling origin expandido**, no k-fold. Un k-fold
      sobre partidos ordenados por fecha vuelve a meter futuro en el
      entrenamiento, y el default de FLAML es justamente un CV aleatorio, así
      que el splitter se le pasa explícito. `RollingOrigin(initial=600,
      assess=100, skip=100)` replica `rsample::rolling_origin`: en `rsample` el
      paso entre bloques es `skip + 1`, así que los bloques de entrenamiento
      terminan en las filas 600, 701 y 802, y cada uno se valida con los 100
      partidos siguientes. Son 3 bloques de 100 partidos. Las filas vienen
      ordenadas por fecha desde R.
    - **Búsqueda:** 500 iteraciones y semilla fija. Sin presupuesto de tiempo,
      que haría que el resultado dependiera de la velocidad de la máquina. Al
      terminar, FLAML reajusta la configuración ganadora con todo el train.
    - **Clase predicha:** con umbral 0,5, igual que en la logística. El umbral
      que maximizaría el accuracy en la validación se muestra sólo como señal
      de calibración: en un modelo calibrado da cerca de 0,5. Las métricas de
      probabilidad (log-loss, Brier) no dependen del umbral.

    Las predicciones quedan en `predicciones/2c/validacion/`, como
    `pred_FLAML_arboles.csv`. Se llama así y no `XGBoost` porque puede ganar
    cualquiera de las tres familias: cuál ganó y con qué configuración queda
    escrito en las notas.
    """)
    return


@app.cell
def _():
    SEMILLA = 2026
    MAX_ITER = 500
    ESTIMADORES = ["xgboost", "lgbm", "rf"]

    # Lo que no es variable del modelo: identificadores, los dos targets y los equipos
    NO_X = ["id_match", "fecha", "temporada", "jornada", "victoria_local",
            "dif_puntos_final", "equipo_local", "equipo_visitante"]
    return ESTIMADORES, MAX_ITER, NO_X, SEMILLA


@app.cell(hide_code=True)
def _(np):
    class RollingOrigin:
        """Folds expandidos: cada bloque entrena con todo el pasado y valida con
        los `assess` partidos siguientes. Replica rsample::rolling_origin(initial,
        assess, skip, cumulative = TRUE), donde el paso entre folds es `skip + 1`.

        Corta por posición de fila, así que las filas tienen que venir ordenadas
        por fecha.

        Dos detalles de FLAML: llama a get_n_splits() sin argumentos (por eso la
        cantidad de filas se guarda en el constructor) y, si el splitter no tiene
        un atributo `shuffle`, asume True y permuta las filas de entrenamiento.
        """

        def __init__(self, initial, assess, skip, n_filas):
            self.initial = initial
            self.assess = assess
            self.paso = skip + 1
            self.n_filas = n_filas
            self.shuffle = False

        def _cortes(self, n):
            fin = self.initial
            while fin + self.assess <= n:
                yield fin
                fin += self.paso

        def get_n_splits(self, X=None, y=None, groups=None):
            n = len(X) if X is not None else self.n_filas
            return sum(1 for _ in self._cortes(n))

        def split(self, X, y=None, groups=None):
            for fin in self._cortes(len(X)):
                yield np.arange(0, fin), np.arange(fin, fin + self.assess)

    return (RollingOrigin,)


@app.cell
def _(mo):
    mo.md("""
    ## Corte 2C: datos al final del 2do cuarto
    """)
    return


@app.cell
def _(NO_X, RollingOrigin, tesina):
    # 1. Cargar. FLAML no acepta polars: X e y le llegan como pandas.
    train_2c, valid_2c, _ = tesina.cargar_split(2)
    X_train_2c = train_2c.drop(NO_X).to_pandas()
    X_valid_2c = valid_2c.drop(NO_X).to_pandas()
    cv_2c = RollingOrigin(initial=600, assess=100, skip=100, n_filas=len(X_train_2c))
    return X_train_2c, X_valid_2c, cv_2c, train_2c, valid_2c


@app.cell
def _(
    AutoML,
    ESTIMADORES,
    MAX_ITER,
    SEMILLA,
    X_train_2c,
    cv_2c,
    mo,
    train_2c,
):
    # 2. Ajustar: FLAML busca por log-loss, validando con el rolling origin del train
    flaml_2c = AutoML()
    flaml_2c.fit(
        X_train=X_train_2c, y_train=train_2c["victoria_local"].to_pandas(),
        task="classification", metric="log_loss",
        estimator_list=ESTIMADORES,
        eval_method="cv", split_type=cv_2c,
        max_iter=MAX_ITER, seed=SEMILLA, verbose=0,
    )
    config_2c = {k: v if isinstance(v, int) else round(float(v), 4)
                 for k, v in flaml_2c.best_config.items()}

    mo.md(f"""
    ### Ajuste: {len(train_2c)} partidos de train, {X_train_2c.shape[1]} variables

    Rolling origin: {cv_2c.get_n_splits(X_train_2c)} bloques de 100 partidos.
    Ganó **{flaml_2c.best_estimator}**; su log-loss medio en esos bloques
    (partidos del train, no de la validación) fue {flaml_2c.best_loss:.5f}.

    Configuración: `{config_2c}`
    """)
    return config_2c, flaml_2c


@app.cell
def _(X_valid_2c, flaml_2c, mo, pl, tesina, valid_2c):
    # Predecir y evaluar sobre la validación
    prob_2c = flaml_2c.predict_proba(X_valid_2c)[:, 1]
    pred_2c = (prob_2c >= 0.5).astype(int)
    umbral_senal_2c = tesina.umbral_optimo(valid_2c["victoria_local"], prob_2c)

    mo.vstack([
        mo.md(f"### Evaluación en la validación: {len(valid_2c)} partidos, del "
              f"{valid_2c['fecha'].min():%Y-%m-%d} al {valid_2c['fecha'].max():%Y-%m-%d}"),
        mo.md(f"Clase predicha con umbral 0,5. El umbral que maximizaría el accuracy en la "
              f"validación es {umbral_senal_2c:.2f}: en un modelo calibrado, cerca de 0,5."),
        tesina.evaluar("FLAML_arboles", valid_2c["victoria_local"], pred_2c, prob_2c)
        .select("modelo", "log_loss", "brier", pl.exclude("modelo", "log_loss", "brier"))
        .with_columns(pl.col(pl.Float64).round(5)),
    ])
    return pred_2c, prob_2c, umbral_senal_2c


@app.cell
def _(
    ESTIMADORES,
    MAX_ITER,
    SEMILLA,
    X_train_2c,
    config_2c,
    cv_2c,
    flaml_2c,
    pred_2c,
    prob_2c,
    tesina,
    umbral_senal_2c,
    valid_2c,
):
    # Guardar las predicciones
    tesina.guardar_predicciones(
        "FLAML_arboles", 2, valid_2c, pred_2c, prob_2c,
        notas=f"FLAML sobre {', '.join(ESTIMADORES)}, tuneo por log-loss con rolling "
              f"origin expandido en el train ({cv_2c.get_n_splits(X_train_2c)} bloques de "
              f"100 partidos; initial=600, assess=100, skip=100); {MAX_ITER} iteraciones, "
              f"semilla {SEMILLA}. Ganó {flaml_2c.best_estimator} con {config_2c}. "
              f"Variables: las {X_train_2c.shape[1]} numéricas de la matriz (todas menos "
              f"identificadores, targets y equipos). Umbral 0,5; el que maximizaría el "
              f"accuracy en la validación es {umbral_senal_2c:.2f}.",
    )
    return


if __name__ == "__main__":
    app.run()
