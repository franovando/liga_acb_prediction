import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import altair as alt
    import marimo as mo
    import polars as pl

    import tesina
    return alt, mo, pl, tesina


@app.cell(hide_code=True)
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
    la calibración de las probabilidades, no el accuracy. Después, el gráfico
    de calibración de los modelos que dan probabilidades y, al final, la
    evaluación de la distribución predictiva de los modelos de la diferencia
    de puntos final.
    """)
    return


@app.cell(hide_code=True)
def _():
    # Los modelos que no miran el partido
    ANTES_DEL_PARTIDO = ["MB0_siempre_local", "MB1_mejor_posicionado"]
    return (ANTES_DEL_PARTIDO,)


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(alt, mo, pl, tesina):
    # Calibración en la validación: los partidos se ordenan por la probabilidad
    # predicha y se parten en 10 grupos de igual cantidad. evaluar_corrida() ya
    # dejó afuera las predicciones que no son de la validación.
    _resultados, _ = tesina.evaluar_corrida(2)
    _modelos = (_resultados.filter(pl.col("log_loss").is_not_nan())
                .sort("log_loss")["modelo"].to_list()) if len(_resultados) else []

    _grupos = [
        pl.read_csv(tesina.RAIZ / "predicciones" / "2c" / "validacion" / f"pred_{_m}.csv")
        .sort("prob_local", "id_match")
        .with_row_index("orden")
        .with_columns(grupo=pl.col("orden") * 10 // pl.len() + 1)
        .group_by("grupo")
        .agg(partidos=pl.len(),
             gano_local=(pl.col("y_real") == "local").sum(),
             predicha=pl.col("prob_local").mean())
        .with_columns(modelo=pl.lit(_m),
                      observada=pl.col("gano_local") / pl.col("partidos"))
        for _m in _modelos
    ]

    if _grupos:
        calibracion = pl.concat(_grupos).sort("modelo", "grupo")

        _x = alt.X("predicha:Q", title="Probabilidad predicha (media del grupo)",
                   scale=alt.Scale(domain=[0, 1]))
        _detalle = [alt.Tooltip("grupo:O", title="Grupo"),
                    alt.Tooltip("partidos:Q", title="Partidos"),
                    alt.Tooltip("predicha:Q", title="Predicha", format=".3f"),
                    alt.Tooltip("observada:Q", title="Observada", format=".3f")]
        _diagonal = (alt.Chart(pl.DataFrame({"x": [0, 1], "y": [0, 1]}))
                     .mark_line(strokeDash=[4, 4], color="gray").encode(x="x:Q", y="y:Q"))
        _puntos = alt.Chart().mark_circle(size=70, opacity=1, color="#4c78a8").encode(
            x=_x,
            y=alt.Y("observada:Q", title="Proporción observada de victorias del local",
                    scale=alt.Scale(domain=[0, 1])),
            tooltip=_detalle,
        )
        _grafico = (alt.layer(_diagonal, _puntos, data=calibracion)
                    .properties(width=300, height=300)
                    .facet(facet=alt.Facet("modelo:N", title=None, sort=_modelos), columns=2))

    mo.vstack([
        mo.md("""
    ### Calibración en la validación

    Para cada modelo que da probabilidades, los partidos de la validación se
    ordenan por la probabilidad predicha de que gane el local y se parten en
    10 grupos de igual cantidad (32 o 33 partidos). Cada punto es un grupo: en
    el eje horizontal, la probabilidad media que le dio el modelo; en el
    vertical, la proporción de esos partidos que ganó el local. En un modelo
    calibrado los puntos caen sobre la diagonal.
    """),
        _grafico,
        calibracion.select("modelo", "grupo", "partidos", "predicha", "observada")
        .with_columns(pl.col(pl.Float64).round(5)),
    ] if _grupos else [mo.md("### Calibración en la validación\n\n"
                             "_No hay predicciones con probabilidad guardadas._")])
    return


@app.cell(hide_code=True)
def _(alt, mo, pl, tesina):
    # Diferencia de puntos final: la distribución predictiva de los modelos que
    # la guardaron con guardar_predictiva(), evaluada con evaluar_predictiva()
    _, _valid, _ = tesina.cargar_split(2)
    _resultados, _pit, _salteados = tesina.evaluar_predictiva(2)

    _bloques = [mo.md(f"""
    ## Diferencia de puntos final

    ### Distribución predictiva en la validación

    {len(_valid)} partidos, del {_valid['fecha'].min():%Y-%m-%d} al
    {_valid['fecha'].max():%Y-%m-%d}. Sólo los modelos que dan una distribución
    para la diferencia final (local − visita, con la prórroga incluida); su
    probabilidad de victoria está además en las tablas de arriba, junto a los
    demás modelos.

    - `log_dens_pred`: promedio del log de la densidad predictiva en la
      diferencia observada. Es el log-loss de una respuesta continua con el
      signo cambiado: más alto es mejor. La tabla va ordenada por esta columna.
    - `cobertura_50` y `cobertura_90`: proporción de partidos cuya diferencia
      final cae en el intervalo predictivo central del 50 % y del 90 %. Con una
      predictiva calibrada, cerca de 0,5 y 0,9; más, si es demasiado ancha, y
      menos, si es demasiado angosta.
    """)]

    if len(_resultados):
        _bloques.append(_resultados.sort("log_dens_pred", descending=True)
                        .with_columns(pl.col(pl.Float64).round(5)))

        # PIT: para cada partido, la proporción de simulaciones por debajo de
        # la diferencia observada, en 10 intervalos de 0,1. Con una predictiva
        # calibrada, cada barra tiene cerca de la décima parte de los partidos
        # (la línea punteada)
        _modelos = _resultados.sort("log_dens_pred", descending=True)["modelo"].to_list()
        _barras = (alt.Chart(_pit)
                   .mark_bar(color="#4c78a8")
                   .encode(x=alt.X("pit:Q", bin=alt.Bin(extent=[0, 1], step=0.1), title="PIT",
                                   axis=alt.Axis(values=[0, 0.2, 0.4, 0.6, 0.8, 1], format=".1f")),
                           y=alt.Y("count():Q", title="Partidos"),
                           tooltip=[alt.Tooltip("count():Q", title="Partidos")]))
        _referencia = (alt.Chart(pl.DataFrame({"y": [len(_valid) / 10]}))
                       .mark_rule(strokeDash=[4, 4], color="gray")
                       .encode(y="y:Q"))
        _bloques.append(mo.md("""
    ### PIT

    Para cada partido, la proporción de las simulaciones de la predictiva que
    quedaron por debajo de la diferencia final observada. Si la predictiva está
    calibrada, esos valores son uniformes entre 0 y 1: cada barra, cerca de la
    línea punteada. Con forma de U, la predictiva es demasiado angosta; con
    forma de joroba, demasiado ancha; inclinada hacia un lado, está corrida.
    """))
        _bloques.append(alt.layer(_barras, _referencia, data=_pit)
                        .properties(width=300, height=220)
                        .facet(facet=alt.Facet("modelo:N", title=None, sort=_modelos),
                               columns=2))
    else:
        _bloques.append(mo.md("_No hay predictivas guardadas._"))

    if len(_salteados):
        _bloques.append(mo.md("**Predictivas salteadas** (no son comparables con el resto):"))
        _bloques.append(_salteados)

    mo.vstack(_bloques)
    return


if __name__ == "__main__":
    app.run()
