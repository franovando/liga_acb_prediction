import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import polars as pl

    import tesina
    return mo, pl, tesina


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Modelos base

    Los modelos contra los que se compara todo lo demás. Son reglas fijas: no
    se ajustan, se aplican directo sobre la validación.

    Primero los que no miran el partido:

    - **MB0**: siempre gana el local.
    - **MB1**: gana el mejor posicionado, el de mayor proporción de victorias
      en sus últimos 34 partidos de fase regular (una fase regular entera),
      contando los de la temporada anterior mientras no haya 34 en la actual.
      Es la fuerza previa por victorias de `etl/05_datos.R`
      (`fuerza_victorias`): a un recién ascendido, cada partido que le falta
      le vale lo de los cinco peores de la temporada anterior. Si hay empate,
      el local.

    Después los que usan lo que pasó hasta el corte:

    - **MB2**: gana el que lidera al corte; si hay empate, el local.
    - **MB3**: los cuatro factores de Oliver con sus pesos originales. Cada
      equipo tiene un puntaje, 0,40·eFG − 0,25·TOV + 0,20·ORB + 0,15·FTR, con
      los factores en puntos porcentuales y acumulados hasta el corte (las
      pérdidas restan). Gana el de puntaje mayor; si hay empate, el local.
      Comparar los dos puntajes ya incluye la defensa: los factores de la
      visita son los factores defensivos del local.
    - **MB4**: votación de MB0, MB1 y MB2: gana el local si lo dan ganador al
      menos dos de los tres. Como MB0 vota siempre local, es lo mismo que
      "gana el local si va ganando al corte o si es el mejor posicionado".

    El corte es el final del 2do cuarto: la diferencia al corte es
    `dif_puntos_Q2`.

    Las predicciones quedan en `predicciones/2c/validacion/`. La comparación con el
    resto de los modelos está en el informe, `resultados.py`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Corte 2C: datos al final del 2do cuarto
    """)
    return


@app.cell(hide_code=True)
def _(tesina):
    # 1. Cargar. Los modelos base no se ajustan, así que sólo hace falta la validación.
    _, valid_2c, _ = tesina.cargar_split(2)
    return (valid_2c,)


@app.cell(hide_code=True)
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
        MB1_mejor_posicionado=pl.col("fuerza_victorias_local") >= pl.col("fuerza_victorias_visita"),
        MB2_lider=pl.col("dif_puntos_Q2") >= 0,
        MB3_cuatro_factores=_puntaje_local >= _puntaje_visita,
    ).with_columns(
        # MB4: al menos dos de MB0, MB1 y MB2 dan ganador al local
        MB4_votacion=(pl.col("MB0_siempre_local").cast(pl.Int8)
                      + pl.col("MB1_mejor_posicionado").cast(pl.Int8)
                      + pl.col("MB2_lider").cast(pl.Int8)) >= 2,
    )
    return (pred_2c,)


@app.cell(hide_code=True)
def _(pl, pred_2c, tesina, valid_2c):
    # 3. Evaluar sobre la validación
    pl.concat([
        tesina.evaluar(_modelo, valid_2c["victoria_local"], pred_2c[_modelo])
        for _modelo in pred_2c.columns
    ]).with_columns(pl.col(pl.Float64).round(5))
    return


@app.cell(hide_code=True)
def _(pred_2c, tesina, valid_2c):
    # Guardar las predicciones
    tesina.guardar_predicciones(
        "MB0_siempre_local", 2, valid_2c, pred_2c["MB0_siempre_local"],
        notas="Predice que gana el local en todos los partidos. No usa variables.",
    )
    tesina.guardar_predicciones(
        "MB1_mejor_posicionado", 2, valid_2c, pred_2c["MB1_mejor_posicionado"],
        notas="Gana el de mayor proporción de victorias en sus últimos 34 partidos "
              "de fase regular, contando los de la temporada anterior (fuerza previa "
              "por victorias de etl/05_datos.R); empate va al local. Variables: "
              "fuerza_victorias_local, fuerza_victorias_visita.",
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
    tesina.guardar_predicciones(
        "MB4_votacion", 2, valid_2c, pred_2c["MB4_votacion"],
        notas="Votación de MB0, MB1 y MB2: gana el local si lo dan ganador al menos "
              "dos de los tres, o sea, si va ganando al corte o si es el mejor "
              "posicionado (MB0 vota siempre local). Variables: dif_puntos_Q2, "
              "fuerza_victorias_local, fuerza_victorias_visita.",
    )
    return


if __name__ == "__main__":
    app.run()
