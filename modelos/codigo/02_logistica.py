import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    import statsmodels.formula.api as smf
    from scipy import stats

    import tesina

    return mo, np, pl, smf, stats, tesina


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Regresión logística

    Una sola regresión logística, con **todas las variables definidas de
    antemano y sin selección**. Las variables se eligieron en términos del
    problema, antes de mirar la validación:

    | Grupo | Variables |
    |---|---|
    | Marcador al corte | `dif_puntos_Q2`, `total_puntos_Q2_c`, `gano_Q1` |
    | Contexto | `es_playoff`, `partidos_jugados_prom` |
    | Cuatro factores, sin eFG% | `dif_tov`, `dif_orb`, `dif_ftr` |
    | Fuerza previa | `dif_fuerza_margen` |

    - Lo que es de cada equipo entra en **diferencia, local − visita**, salvo
      los partidos jugados, que miden el momento de la temporada y entran como
      el **promedio de los dos equipos** (`partidos_jugados_prom`). Antes de
      ajustar se testea que eso sea razonable (test de simetría).
    - **eFG% queda afuera**: con el marcador en el modelo no mediría tirar
      mejor sino cuántos tiros hicieron falta para llegar a ese marcador, y
      está muy correlacionado con `dif_puntos_Q2`.
    - **Sin variables muy correlacionadas entre sí**: si dos miden casi lo
      mismo, sus coeficientes se estiman mal y no se pueden leer por separado.
      Por eso, de las dos medidas de fuerza previa (`etl/05_datos.R`) entra
      sólo el margen medio, y los partidos jugados de los dos equipos, casi
      siempre iguales, entran promediados.
    - Del 1er cuarto entra **quién lo ganó**, no por cuánto: `gano_Q1` vale 1
      si lo ganó el local, −1 si lo ganó la visita y 0 si empataron. Los
      puntos del 1er cuarto ya están en `dif_puntos_Q2`; la pregunta es si, a
      igual marcador al descanso, haber ganado el 1er cuarto agrega algo. Su
      coeficiente β compara ganarlo con empatarlo; ganarlo contra perderlo
      vale 2β (OR = e^(2β)).
    - `total_puntos_Q2` entra **centrado en la media del train**
      (`total_puntos_Q2_c`). Centrar no cambia las predicciones ni los tests,
      sólo cómo se lee el intercepto.
    - Sin el box score crudo, sin equipos y sin interacciones.

    **Por qué sin selección.** Quedarse con las variables que dan
    significativas sobre los mismos datos sesga sus coeficientes (un estimador
    insesgado deja de serlo si se lo conserva sólo cuando da significativo) y
    vuelve optimistas sus p-valores. Con pocas variables para la cantidad de
    partidos del train, no hace falta achicar el modelo para estimarlo bien.

    Las comparaciones entre modelos anidados son tests de razón de
    verosimilitud, con `rv()`: G = 2 (log L del modelo grande − log L del
    chico), con tantos grados de libertad como parámetros de diferencia.

    **Clase predicha:** umbral de 0,5. El umbral que maximizaría el accuracy
    en la validación (`umbral_optimo()`) se muestra sólo como señal de
    calibración: en un modelo calibrado da cerca de 0,5. Los modelos se
    comparan por log-loss, que no depende del umbral. El gráfico de
    calibración está en `resultados.py`.

    El modelo se evalúa en la **validación** (la temporada 2024-25 entera).
    El **test**, la temporada 2025-26, no se usa acá: queda para el final,
    para el modelo que se elija.

    Las predicciones quedan en `predicciones/2c/validacion/pred_logistica.csv`.
    Los modelos base están en `01_modelos_base.py`.
    """)
    return


@app.cell(hide_code=True)
def _(np, pl):
    def tabla_coeficientes(modelo):
        """Coeficientes de una logística de statsmodels: estimación, error
        estándar y odds ratio con su IC 95 %. statsmodels los devuelve en
        pandas: se pasan a numpy apenas salen."""
        ic = np.exp(modelo.conf_int().to_numpy())
        return pl.DataFrame({
            "variable": modelo.model.exog_names,
            "coef": modelo.params.to_numpy(),
            "ee": modelo.bse.to_numpy(),
            "OR": np.exp(modelo.params.to_numpy()),
            "OR_ic_inf": ic[:, 0],
            "OR_ic_sup": ic[:, 1],
        }).with_columns(pl.col(pl.Float64).round(4))

    return (tabla_coeficientes,)


@app.cell(hide_code=True)
def _(stats):
    def rv(modelo_grande, modelo_chico):
        """Test de razón de verosimilitud entre dos logísticas anidadas,
        ajustadas sobre los mismos datos: estadístico G, grados de libertad y
        p-valor."""
        g = 2 * (modelo_grande.llf - modelo_chico.llf)
        gl = int(modelo_grande.df_model - modelo_chico.df_model)
        return g, gl, stats.chi2.sf(g, gl)

    return (rv,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Corte 2C: datos al final del 2do cuarto
    """)
    return


@app.cell(hide_code=True)
def _(pl, tesina):
    # 1. Cargar. El ritmo se centra con la media del train, también en la
    # validación. Lo que es de cada equipo, en diferencia local − visita; los
    # partidos jugados, en promedio. Del 1er cuarto, quién lo ganó (1, −1 o 0).
    _train, _valid, _ = tesina.cargar_split(2)
    media_ritmo_2c = _train["total_puntos_Q2"].mean()
    _nuevas = [
        (pl.col("total_puntos_Q2") - media_ritmo_2c).alias("total_puntos_Q2_c"),
        *[(pl.col(f"local_{_f}") - pl.col(f"visita_{_f}")).alias(f"dif_{_f}")
          for _f in ["tov", "orb", "ftr"]],
        *[(pl.col(f"fuerza_{_f}_local") - pl.col(f"fuerza_{_f}_visita")).alias(f"dif_fuerza_{_f}")
          for _f in ["margen", "victorias"]],
        ((pl.col("partidos_jugados_local") + pl.col("partidos_jugados_visita")) / 2)
        .alias("partidos_jugados_prom"),
        pl.col("dif_puntos_Q1").sign().alias("gano_Q1"),
    ]
    train_2c = _train.with_columns(_nuevas)
    valid_2c = _valid.with_columns(_nuevas)

    variables_2c = ["dif_puntos_Q2", "total_puntos_Q2_c", "gano_Q1", "es_playoff",
                    "partidos_jugados_prom", "dif_tov", "dif_orb", "dif_ftr", "dif_fuerza_margen"]
    return media_ritmo_2c, train_2c, valid_2c, variables_2c


@app.cell(hide_code=True)
def _(mo, pl, rv, smf, train_2c):
    # Test de simetría: las variables de cada equipo por separado contra su
    # diferencia (los partidos jugados, contra su promedio), con el marcador en
    # el modelo
    _base = "victoria_local ~ dif_puntos_Q2 + total_puntos_Q2_c + es_playoff"
    _pruebas = [
        ("pérdidas, rebote y libres",
         ["local_tov", "visita_tov", "local_orb", "visita_orb", "local_ftr", "visita_ftr"],
         ["dif_tov", "dif_orb", "dif_ftr"]),
        ("fuerza (margen)", ["fuerza_margen_local", "fuerza_margen_visita"], ["dif_fuerza_margen"]),
        ("partidos jugados", ["partidos_jugados_local", "partidos_jugados_visita"],
         ["partidos_jugados_prom"]),
    ]
    _filas = []
    for _nombre, _por_equipo, _combinadas in _pruebas:
        _grande = smf.logit(_base + " + " + " + ".join(_por_equipo), data=train_2c).fit(disp=0)
        _chico = smf.logit(_base + " + " + " + ".join(_combinadas), data=train_2c).fit(disp=0)
        _g, _gl, _p = rv(_grande, _chico)
        _filas.append({"variables": _nombre, "G": _g, "gl": _gl, "p_valor": _p})

    mo.vstack([
        mo.md("""
    ### Test de simetría

    Las variables de cada equipo por separado contra su diferencia, local −
    visita, sobre un modelo con el marcador (`dif_puntos_Q2`,
    `total_puntos_Q2_c` y `es_playoff`), con razón de verosimilitud. Usar la
    diferencia supone que cada variable pesa lo mismo para los dos equipos,
    con el signo cambiado; si el test no rechaza, es razonable. Los partidos
    jugados se prueban contra su promedio, que supone que pesan lo mismo y con
    el mismo signo: miden en qué momento de la temporada se juega el partido,
    no algo de cada equipo.
    """),
        pl.DataFrame(_filas).with_columns(pl.col(pl.Float64).round(4)),
        mo.md("Correlación en el train entre las dos medidas de fuerza, en diferencias: "
              f"{train_2c.select(pl.corr('dif_fuerza_margen', 'dif_fuerza_victorias')).item():.3f}; "
              "entre los partidos jugados de los dos equipos: "
              f"{train_2c.select(pl.corr('partidos_jugados_local', 'partidos_jugados_visita')).item():.3f}. "
              "Por eso entran sólo la fuerza por margen y el promedio de los partidos jugados."),
    ])
    return


@app.cell(hide_code=True)
def _(mo, smf, tabla_coeficientes, train_2c, variables_2c):
    # 2. Ajustar sobre el train, con todas las variables
    formula_2c = "victoria_local ~ " + " + ".join(variables_2c)
    logistica_2c = smf.logit(formula_2c, data=train_2c).fit(disp=0)
    _visitantes = int((1 - train_2c["victoria_local"]).sum())

    mo.vstack([
        mo.md(f"""
    ### Modelo

    `{formula_2c}`

    Ajustado sobre los {len(train_2c)} partidos del train: {_visitantes} victorias
    visitantes (la clase menos frecuente) para {len(variables_2c)} variables,
    {_visitantes / len(variables_2c):.0f} por variable.
    """),
        tabla_coeficientes(logistica_2c),
    ])
    return (logistica_2c,)


@app.cell(hide_code=True)
def _(logistica_2c, mo, pl, tesina, valid_2c):
    # 3. Predecir y evaluar sobre la validación. La clase sale de un umbral de 0,5
    prob_2c = logistica_2c.predict(valid_2c).to_numpy()
    pred_2c = (prob_2c >= 0.5).astype(int)
    umbral_senal_2c = tesina.umbral_optimo(valid_2c["victoria_local"], prob_2c)

    mo.vstack([
        mo.md(f"### Evaluación en la validación: {len(valid_2c)} partidos, del "
              f"{valid_2c['fecha'].min():%Y-%m-%d} al {valid_2c['fecha'].max():%Y-%m-%d}"),
        mo.md(f"Clase predicha con umbral 0,5. El umbral que maximizaría el accuracy en la "
              f"validación es {umbral_senal_2c:.2f}: en un modelo calibrado, cerca de 0,5."),
        tesina.evaluar("logistica", valid_2c["victoria_local"], pred_2c, prob_2c)
        .select("modelo", "log_loss", "brier", pl.exclude("modelo", "log_loss", "brier"))
        .with_columns(pl.col(pl.Float64).round(5)),
    ])
    return pred_2c, prob_2c, umbral_senal_2c


@app.cell(hide_code=True)
def _(
    media_ritmo_2c,
    pred_2c,
    prob_2c,
    tesina,
    umbral_senal_2c,
    valid_2c,
    variables_2c,
):
    # Guardar las predicciones
    tesina.guardar_predicciones(
        "logistica", 2, valid_2c, pred_2c, prob_2c,
        notas="Regresión logística (statsmodels, máxima verosimilitud) con todas las variables "
              f"definidas de antemano, sin selección. Variables: {', '.join(variables_2c)}. "
              "Las dif_ son "
              "local − visita; dif_fuerza_margen sale de la fuerza previa por margen medio de "
              "etl/05_datos.R (los últimos 34 partidos de fase regular antes del partido, contando "
              "los de la temporada anterior). partidos_jugados_prom es el promedio de partidos_jugados_local "
              "y partidos_jugados_visita. gano_Q1 es 1 si el local ganó el 1er cuarto, −1 si lo "
              "ganó la visita y 0 si empataron. total_puntos_Q2_c es total_puntos_Q2 centrado en la "
              "media del "
              f"train ({media_ritmo_2c:.1f}). Umbral 0,5; el que maximizaría el accuracy en la "
              f"validación es {umbral_senal_2c:.2f}.",
    )
    return


if __name__ == "__main__":
    app.run()
