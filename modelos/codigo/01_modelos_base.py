import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import polars as pl

    import tesina
    return mo, pl, tesina


@app.cell
def _(mo):
    mo.md("""
    # Modelos base

    Los modelos contra los que se compara todo lo demás. Son reglas fijas: no
    se ajustan, se aplican directo sobre la validación.

    Primero los que no miran el partido:

    - **MB0**: siempre gana el local.
    - **MB1**: gana el mejor posicionado, el de mejor porcentaje de victorias
      en la fase regular de la temporada anterior; un recién ascendido cuenta
      como el peor. Si hay empate, el local.

    Después los que usan lo que pasó hasta el corte:

    - **MB2**: gana el que lidera al corte; si hay empate, el local.
    - **MB3**: los cuatro factores de Oliver con sus pesos originales. Cada
      equipo tiene un puntaje, 0,40·eFG − 0,25·TOV + 0,20·ORB + 0,15·FTR, con
      los factores en puntos porcentuales y acumulados hasta el corte (las
      pérdidas restan). Gana el de puntaje mayor; si hay empate, el local.
      Comparar los dos puntajes ya incluye la defensa: los factores de la
      visita son los factores defensivos del local.

    El corte es el final del 2do cuarto: la diferencia al corte es
    `dif_puntos_Q2`.

    Las predicciones quedan en `predicciones/2c/validacion/`. La comparación con el
    resto de los modelos está en el informe, `resultados.py`.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Corte 2C: datos al final del 2do cuarto
    """)
    return


@app.cell
def _(tesina):
    # 1. Cargar. Los modelos base no se ajustan, así que sólo hace falta la validación.
    _, valid_2c, _ = tesina.cargar_split(2)
    return (valid_2c,)


@app.cell
def _(pl, valid_2c):
    # 2. Aplicar las reglas sobre la validación: una columna por modelo base,
    # True si predice que gana el local
    # MB3: puntaje de Oliver de cada equipo, con los factores en puntos porcentuales
    _puntaje_local = (0.40 * pl.col("local_efg") - 0.25 * pl.col("local_tov")
                      + 0.20 * pl.col("local_orb") + 0.15 * pl.col("local_ftr"))
    _puntaje_visita = (0.40 * pl.col("visita_efg") - 0.25 * pl.col("visita_tov")
                       + 0.20 * pl.col("visita_orb") + 0.15 * pl.col("visita_ftr"))
    pred_2c = valid_2c.select(
        MB0_siempre_local=pl.lit(True),
        # MB1: el recién ascendido (sin temporada anterior, null) cuenta como 0
        MB1_mejor_posicionado=(pl.col("pct_victorias_anterior_local").fill_null(0)
                               >= pl.col("pct_victorias_anterior_visita").fill_null(0)),
        MB2_lider=pl.col("dif_puntos_Q2") >= 0,
        MB3_cuatro_factores=_puntaje_local >= _puntaje_visita,
    )
    return (pred_2c,)


@app.cell
def _(pl, pred_2c, tesina, valid_2c):
    # 3. Evaluar sobre la validación
    pl.concat([
        tesina.evaluar(_modelo, valid_2c["victoria_local"], pred_2c[_modelo])
        for _modelo in pred_2c.columns
    ]).with_columns(pl.col(pl.Float64).round(5))
    return


@app.cell
def _(pred_2c, tesina, valid_2c):
    # Guardar las predicciones
    tesina.guardar_predicciones(
        "MB0_siempre_local", 2, valid_2c, pred_2c["MB0_siempre_local"],
        notas="Predice que gana el local en todos los partidos. No usa variables.",
    )
    tesina.guardar_predicciones(
        "MB1_mejor_posicionado", 2, valid_2c, pred_2c["MB1_mejor_posicionado"],
        notas="Gana el de mejor porcentaje de victorias en la fase regular de la "
              "temporada anterior; recién ascendido cuenta como 0; empate va al local. "
              "Variables: pct_victorias_anterior_local, pct_victorias_anterior_visita.",
    )
    tesina.guardar_predicciones(
        "MB2_lider", 2, valid_2c, pred_2c["MB2_lider"],
        notas="Gana quien lidera al corte; empate va al local. Variable: dif_puntos_Q2.",
    )
    tesina.guardar_predicciones(
        "MB3_cuatro_factores", 2, valid_2c, pred_2c["MB3_cuatro_factores"],
        notas="Gana el de mayor puntaje de Oliver, 0,40·eFG − 0,25·TOV + 0,20·ORB + "
              "0,15·FTR, con los factores de cada equipo en puntos porcentuales y "
              "acumulados hasta el corte; empate va al local. Variables: local_efg, "
              "local_tov, local_orb, local_ftr, visita_efg, visita_tov, visita_orb, "
              "visita_ftr.",
    )
    return


if __name__ == "__main__":
    app.run()
