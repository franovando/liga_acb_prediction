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
    # Resultados

    Métricas de todos los modelos que dejaron predicciones en
    `predicciones/2c/`, calculadas en el momento con `evaluar()` de
    `tesina.py`. Este informe no entrena nada: sólo lee predicciones.

    - **Validación:** la temporada 2024-25 entera. Ahí se comparan los
      modelos: todos se evalúan sobre los mismos partidos.
    - **Test:** la temporada 2025-26 entera. Sólo tiene predicciones el modelo
      elegido, evaluado una vez al final.

    Primero van los modelos que no miran el partido (MB0 y MB1) y después los
    que usan lo que pasó hasta el corte, el final del 2do cuarto.

    Las features pueden variar de un modelo a otro; cada uno dice cuáles usa en
    la columna `notas`. Cada tabla va ordenada por log-loss: lo que importa es
    la calibración de las probabilidades, no el accuracy.
    """)
    return


@app.cell
def _():
    # Los modelos que no miran el partido
    ANTES_DEL_PARTIDO = ["MB0_siempre_local", "MB1_mejor_posicionado"]
    return (ANTES_DEL_PARTIDO,)


@app.cell
def _(ANTES_DEL_PARTIDO, mo, pl, tesina):
    _, _valid, _ = tesina.cargar_split(2)
    _resultados, _ = tesina.evaluar_corrida(2)
    if len(_resultados):
        _resultados = _resultados.filter(pl.col("modelo").is_in(ANTES_DEL_PARTIDO))

    mo.vstack([
        mo.md(f"""
    ## Antes del partido

    ### Validación

    {len(_valid)} partidos, del {_valid['fecha'].min():%Y-%m-%d} al
    {_valid['fecha'].max():%Y-%m-%d} ({int(_valid['victoria_local'].sum())} ganó
    el local, {int((1 - _valid['victoria_local']).sum())} la visita).
    """),
        _resultados.sort("log_loss")
        .select("modelo", "log_loss", "brier", pl.exclude("modelo", "log_loss", "brier"))
        .with_columns(pl.col(pl.Float64).round(5)) if len(_resultados)
        else mo.md("_No hay predicciones guardadas._"),
    ])
    return


@app.cell
def _(ANTES_DEL_PARTIDO, mo, pl, tesina):
    _train, _valid, _test = tesina.cargar_split(2)
    _desde = int(_train["temporada"].min())
    _hasta = int(_train["temporada"].max())

    secciones = [mo.md(f"""
    ## Corte 2C

    - **Train:** {len(_train)} partidos, temporadas
      {_desde}-{str(_desde + 1)[-2:]} a {_hasta}-{str(_hasta + 1)[-2:]}
      (del {_train['fecha'].min():%Y-%m-%d} al {_train['fecha'].max():%Y-%m-%d}).
    """)]

    for _conjunto, _datos, _nombre in (("validacion", _valid, "Validación"),
                                       ("test", _test, "Test")):
        _resultados, _salteados = tesina.evaluar_corrida(2, _conjunto)
        if len(_resultados):
            _resultados = _resultados.filter(~pl.col("modelo").is_in(ANTES_DEL_PARTIDO))
        secciones.append(mo.md(f"""
    ### {_nombre}

    {len(_datos)} partidos, del {_datos['fecha'].min():%Y-%m-%d} al
    {_datos['fecha'].max():%Y-%m-%d} ({int(_datos['victoria_local'].sum())} ganó
    el local, {int((1 - _datos['victoria_local']).sum())} la visita).
    """))
        secciones.append(_resultados.sort("log_loss")
                         .select("modelo", "log_loss", "brier",
                                 pl.exclude("modelo", "log_loss", "brier"))
                         .with_columns(pl.col(pl.Float64).round(5))
                         if len(_resultados)
                         else mo.md("_No hay predicciones guardadas._"))
        if len(_salteados):
            secciones.append(mo.md("**Predicciones salteadas** (no son "
                                   "comparables con el resto):"))
            secciones.append(_salteados)

    mo.vstack(secciones)
    return


if __name__ == "__main__":
    app.run()
