import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    import statsmodels.formula.api as smf
    from scipy import stats

    import tesina

    return mo, np, pl, smf, stats, tesina


@app.cell
def _(mo):
    mo.md("""
    # Regresión logística: selección de variables por el método de Collett

    Una sola regresión logística. Sus variables se eligen con el procedimiento
    de Collett, todo sobre el train:

    1. **Paso 1.** Se ajusta un modelo univariado con todas las variables a evaluar
       y se lo compara con el modelo nulo (sólo intercepto). Siguen las que
       dan **p < 0,2**.
    2. **Paso 2.** Las que siguieron van juntas en un modelo. Se omite cada una
       por turno y se compara con el modelo que la tiene. Sale la de p-valor
       más alto, si da **p ≥ 0,1**, y con las que quedan se vuelve a examinar la
       omisión de cada una. Termina cuando todas dan p < 0,1: en presencia de
       las demás, algunas pueden dejar de ser importantes.
    3. **Paso 3.** Las que no siguieron en el paso 1 se agregan de a una al
       modelo del paso 2 (entra la de p-valor más bajo, si da **p < 0,05**): en
       presencia de las demás pueden resultar importantes. Si entra alguna, se
       vuelve a examinar la omisión de todas como en el paso 2, y se prueba otra
       vez con las que siguen afuera.
    4. **Paso 4.** Se ajusta el modelo final.

    Todas las comparaciones son tests de razón de verosimilitud entre modelos
    anidados, con `rv()`: G = 2 (log L del modelo grande − log L del chico),
    con tantos grados de libertad como parámetros de diferencia. Los pasos 1
    a 3 los hace la función `collett()`.

    **Candidatas**, definidas antes de mirar la validación:

    | Grupo | Candidatas |
    |---|---|
    | Marcador al corte | `dif_puntos_Q2`, `total_puntos_Q2_c`, `dif_puntos_Q1` |
    | Contexto | `es_playoff`, `partidos_jugados_prom` |
    | Cuatro factores, sin eFG% | `dif_tov`, `dif_orb`, `dif_ftr` |
    | Fuerza previa | `dif_fuerza_margen` |

    - Lo que es de cada equipo entra en **diferencia, local − visita**, salvo
      los partidos jugados, que miden el momento de la temporada y entran como
      el **promedio de los dos equipos** (`partidos_jugados_prom`). Antes de la
      selección se testea que eso sea razonable (test de simetría).
    - **eFG% queda afuera**: con el marcador en el modelo no mediría tirar
      mejor sino cuántos tiros hicieron falta para llegar a ese marcador, y
      está muy correlacionado con `dif_puntos_Q2`.
    - **Sin candidatas muy correlacionadas entre sí**: si dos miden casi lo
      mismo, cuál queda en el modelo depende de diferencias chicas de la
      muestra. Por eso, de las dos medidas de fuerza previa (`etl/05_datos.R`)
      entra sólo el margen medio, y los partidos jugados de los dos equipos,
      casi siempre iguales, entran promediados. `dif_puntos_Q1` y
      `dif_puntos_Q2` están correlacionadas, pero mucho menos; entran las dos
      porque la pregunta es justamente si el 1er cuarto agrega algo.
    - `total_puntos_Q2` entra **centrado en la media del train**
      (`total_puntos_Q2_c`). Centrar no cambia las predicciones ni los tests,
      sólo cómo se lee el intercepto.
    - Sin el box score crudo, sin equipos y sin interacciones.

    Los p-valores de la selección son optimistas: las variables se eligieron
    mirando esos mismos datos, así que se leen como descripción y no como
    tests. La validación no interviene en la selección, así que sus métricas
    no tienen ese problema.

    **Clase predicha:** umbral de 0,5. El umbral que maximizaría el accuracy
    en la validación (`umbral_optimo()`) se muestra sólo como señal de
    calibración: en un modelo calibrado da cerca de 0,5. Los modelos se
    comparan por log-loss, que no depende del umbral.

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
        estándar y odds ratio con su IC 95 %."""
        ic = np.exp(modelo.conf_int())
        return pl.DataFrame({
            "variable": modelo.params.index,
            "coef": modelo.params,
            "ee": modelo.bse,
            "OR": np.exp(modelo.params),
            "OR_ic_inf": ic[0],
            "OR_ic_sup": ic[1],
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
def _(pl, rv, smf):
    def collett(candidatas, datos):
        """Pasos 1 a 3 del método de Collett (descriptos arriba) sobre `datos`,
        el train. Devuelve la tabla de cada paso ("paso1", "paso2", "paso3"),
        redondeada a 4 decimales para mostrarla (las decisiones se toman sin
        redondear), las variables que quedan después del paso 2
        ("paso2_quedan") y las del modelo final ("final")."""
        nulo = smf.logit("victoria_local ~ 1", data=datos).fit(disp=0)

        # Paso 1: cada candidata sola contra el modelo nulo (alfa = 0,2)
        paso1 = []
        for v in candidatas:
            modelo = smf.logit(f"victoria_local ~ {v}", data=datos).fit(disp=0)
            g, gl, p = rv(modelo, nulo)
            paso1.append({"variable": v, "G": g, "gl": gl, "p_valor": p,
                          "decision": "sigue" if p < 0.2 else "no sigue"})
        siguen = [f["variable"] for f in paso1 if f["decision"] == "sigue"]

        # Paso 2: se omiten de a una (alfa = 0,1)
        variables = list(siguen)
        paso2 = []
        ronda = 0
        while variables:
            ronda += 1
            grande = smf.logit("victoria_local ~ " + " + ".join(variables),
                               data=datos).fit(disp=0)
            desde = len(paso2)
            for v in variables:
                resto = [w for w in variables if w != v]
                chico = smf.logit("victoria_local ~ " + (" + ".join(resto) or "1"),
                                  data=datos).fit(disp=0)
                g, gl, p = rv(grande, chico)
                paso2.append({"ronda": ronda, "se_omite": v, "G": g, "gl": gl,
                              "p_valor": p, "decision": "queda"})
            peor = max(paso2[desde:], key=lambda f: f["p_valor"])
            if peor["p_valor"] < 0.1:
                break
            peor["decision"] = "sale"
            variables.remove(peor["se_omite"])
        quedan = list(variables)

        # Paso 3: se agregan de a una las que no siguieron (alfa = 0,05)
        afuera = [v for v in candidatas if v not in siguen]
        paso3 = []
        ronda = 0
        while afuera:
            ronda += 1
            chico = smf.logit("victoria_local ~ " + (" + ".join(variables) or "1"),
                              data=datos).fit(disp=0)
            desde = len(paso3)
            for v in afuera:
                grande = smf.logit("victoria_local ~ " + " + ".join(variables + [v]),
                                   data=datos).fit(disp=0)
                g, gl, p = rv(grande, chico)
                paso3.append({"ronda": ronda, "prueba": "se agrega", "variable": v,
                              "G": g, "gl": gl, "p_valor": p, "decision": "no entra"})
            mejor = min(paso3[desde:], key=lambda f: f["p_valor"])
            if mejor["p_valor"] >= 0.05:
                break
            mejor["decision"] = "entra"
            variables.append(mejor["variable"])
            afuera.remove(mejor["variable"])

            # Con la nueva adentro, se vuelve a examinar la omisión de cada una
            while True:
                grande = smf.logit("victoria_local ~ " + " + ".join(variables),
                                   data=datos).fit(disp=0)
                desde = len(paso3)
                for v in variables:
                    resto = [w for w in variables if w != v]
                    chico = smf.logit("victoria_local ~ " + (" + ".join(resto) or "1"),
                                      data=datos).fit(disp=0)
                    g, gl, p = rv(grande, chico)
                    paso3.append({"ronda": ronda, "prueba": "se omite", "variable": v,
                                  "G": g, "gl": gl, "p_valor": p, "decision": "queda"})
                peor = max(paso3[desde:], key=lambda f: f["p_valor"])
                if peor["p_valor"] < 0.1:
                    break
                peor["decision"] = "sale"
                variables.remove(peor["variable"])

        redondear = pl.col(pl.Float64).round(4)
        return {
            "paso1": pl.DataFrame(paso1).with_columns(redondear),
            "paso2": pl.DataFrame(paso2).with_columns(redondear),
            "paso2_quedan": quedan,
            "paso3": pl.DataFrame(paso3).with_columns(redondear),
            "final": variables,
        }

    return (collett,)


@app.cell
def _(mo):
    mo.md("""
    ## Corte 2C: datos al final del 2do cuarto
    """)
    return


@app.cell
def _(pl, tesina):
    # 1. Cargar. El ritmo se centra con la media del train, también en la
    # validación. Lo que es de cada equipo, en diferencia local − visita; los
    # partidos jugados, en promedio.
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
    ]
    train_2c = _train.with_columns(_nuevas)
    valid_2c = _valid.with_columns(_nuevas)

    candidatas_2c = ["dif_puntos_Q2", "total_puntos_Q2_c", "dif_puntos_Q1", "es_playoff",
                     "partidos_jugados_prom", "dif_tov", "dif_orb", "dif_ftr", "dif_fuerza_margen"]
    return candidatas_2c, media_ritmo_2c, train_2c, valid_2c


@app.cell
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


@app.cell
def _(candidatas_2c, collett, mo, train_2c):
    # Pasos 1 a 3 del método de Collett
    seleccion_2c = collett(candidatas_2c, train_2c)
    final_2c = seleccion_2c["final"]

    mo.vstack([
        mo.md("### Paso 1: cada variable sola contra el modelo nulo (α = 0,2)"),
        seleccion_2c["paso1"],
        mo.md("### Paso 2: se omiten de a una (α = 0,1)"),
        seleccion_2c["paso2"],
        mo.md(f"Quedan: `{'`, `'.join(seleccion_2c['paso2_quedan'])}`."),
        mo.md("### Paso 3: se agregan de a una las que no siguieron en el paso 1 "
              "(α = 0,05)"),
        seleccion_2c["paso3"] if len(seleccion_2c["paso3"])
        else mo.md("_Todas siguieron en el paso 1: no hay nada que agregar._"),
        mo.md(f"Modelo final: `{'`, `'.join(final_2c)}`."),
    ])
    return (final_2c,)


@app.cell
def _(final_2c, mo, smf, tabla_coeficientes, train_2c):
    # Paso 4: el modelo final, ajustado sobre el train
    formula_2c = "victoria_local ~ " + " + ".join(final_2c)
    lc_2c = smf.logit(formula_2c, data=train_2c).fit(disp=0)

    mo.vstack([
        mo.md(f"### Paso 4: modelo final\n\n`{formula_2c}`"),
        tabla_coeficientes(lc_2c),
    ])
    return (lc_2c,)


@app.cell
def _(lc_2c, mo, pl, tesina, valid_2c):
    # Predecir y evaluar sobre la validación. La clase sale de un umbral de 0,5
    prob_2c = lc_2c.predict(valid_2c).to_numpy()
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


@app.cell
def _(
    candidatas_2c,
    final_2c,
    media_ritmo_2c,
    pred_2c,
    prob_2c,
    tesina,
    umbral_senal_2c,
    valid_2c,
):
    # Guardar las predicciones
    tesina.guardar_predicciones(
        "logistica", 2, valid_2c, pred_2c, prob_2c,
        notas="Regresión logística (statsmodels, máxima verosimilitud). Variables elegidas en "
              "el train por el método de Collett (α 0,2 / 0,1 / 0,05) entre "
              f"{', '.join(candidatas_2c)}. Variables: {', '.join(final_2c)}. Las dif_ son "
              "local − visita; dif_fuerza_margen sale de la fuerza previa por margen medio de "
              "etl/05_datos.R (los últimos 34 partidos de fase regular antes del partido, contando "
              "los de la temporada anterior). partidos_jugados_prom es el promedio de partidos_jugados_local "
              "y partidos_jugados_visita. total_puntos_Q2_c es total_puntos_Q2 centrado en la "
              "media del "
              f"train ({media_ritmo_2c:.1f}). Umbral 0,5; el que maximizaría el accuracy en la "
              f"validación es {umbral_senal_2c:.2f}.",
    )
    return


if __name__ == "__main__":
    app.run()
