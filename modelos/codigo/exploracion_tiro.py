import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell
def _():
    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl
    import statsmodels.formula.api as smf
    from scipy import stats

    import tesina

    return alt, mo, np, pl, smf, stats, tesina


@app.cell
def _(mo):
    mo.md("""
    # La diferencia al 2C, el resultado y el tiro

    Qué dice la diferencia de puntos al final del 2do cuarto (`dif_puntos_Q2`,
    local − visita; es la suma de los márgenes del 1er y del 2do cuarto) sobre
    el resultado del partido, y cómo se relaciona con el tiro de
    cada equipo en la primera mitad:

    1. **Poder predictivo de la diferencia al 2C**, sobre la victoria del local
       (`victoria_local`) y sobre la diferencia final (`dif_puntos_final`).
    2. **La diferencia al 2C y las estadísticas avanzadas de tiro.**

    **Sólo el train** (temporadas 2021-22 a 2023-24). La validación y el test no
    se miran: si algo de acá termina en un modelo, ese modelo se compara en la
    validación, y explorar sobre ella la gastaría. Las medidas de ajuste son
    dentro de la muestra; con uno o dos parámetros y casi mil partidos el
    sobreajuste es despreciable. Fuera de la muestra, la diferencia sola ya está
    evaluada en la validación como MB2 (el líder), y junto a la fase, las
    pérdidas y la fuerza previa en `logistica` (ver `resultados.py`).

    Es exploratorio: las métricas de tiro se arman acá a partir de las
    columnas del box score de la matriz, y no se guardan predicciones. Si alguna pasa a un modelo, se calcula en
    `etl/05_datos.R`, como el resto de la matriz.
    """)
    return


@app.cell
def _(pl, tesina):
    # 1. Cargar: sólo el train. margen_2T es el margen del 2do tiempo (con
    # prórroga); tramo agrupa la diferencia al 2C.
    _train, _, _ = tesina.cargar_split(2)
    _m = pl.col("dif_puntos_Q2")
    train = _train.with_columns(
        margen_2T=pl.col("dif_puntos_final") - _m,
        tramo=pl.when(_m <= -15).then(pl.lit("≤ −15"))
        .when(_m <= -10).then(pl.lit("−14 a −10"))
        .when(_m <= -5).then(pl.lit("−9 a −5"))
        .when(_m <= -1).then(pl.lit("−4 a −1"))
        .when(_m == 0).then(pl.lit("0"))
        .when(_m <= 4).then(pl.lit("1 a 4"))
        .when(_m <= 9).then(pl.lit("5 a 9"))
        .when(_m <= 14).then(pl.lit("10 a 14"))
        .otherwise(pl.lit("≥ 15")),
    )
    return (train,)


@app.cell
def _(mo, pl, train):
    _tramos = (
        train.group_by("tramo")
        .agg(
            desde=pl.col("dif_puntos_Q2").min(),
            partidos=pl.len(),
            gana_local=pl.col("victoria_local").mean(),
            media_final=pl.col("dif_puntos_final").mean(),
            de_final=pl.col("dif_puntos_final").std(),
            media_2T=pl.col("margen_2T").mean(),
        )
        .sort("desde")
        .drop("desde")
        .with_columns(pl.col(pl.Float64).round(3))
    )

    mo.vstack([
        mo.md(f"""
    ## 1. Poder predictivo de la diferencia al 2C

    Train: {len(train)} partidos, del {train['fecha'].min():%Y-%m-%d} al
    {train['fecha'].max():%Y-%m-%d} ({int(train['victoria_local'].sum())} ganó
    el local, {int((1 - train['victoria_local']).sum())} la visita).

    Por tramo de la diferencia al 2C: proporción de victorias del local, media y
    desvío de la diferencia final y media del margen del 2do tiempo.
    """),
        _tramos,
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ### Victoria del local

    Una logística con la diferencia sola, `victoria_local ~ dif_puntos_Q2`, y la
    regla del líder (MB2: gana el que va ganando; empate, el local). Las
    métricas salen de `evaluar()`, dentro de la muestra; la logística pasa a
    clase con 0,5.
    """)
    return


@app.cell
def _(smf, train):
    logit_q2 = smf.logit("victoria_local ~ dif_puntos_Q2", data=train).fit(disp=0)
    return (logit_q2,)


@app.cell
def _(logit_q2, mo, np, pl, tesina, train):
    _prob = logit_q2.predict(train).to_numpy()
    _ic = np.exp(logit_q2.conf_int())
    _coef = pl.DataFrame({
        "variable": logit_q2.params.index,
        "coef": logit_q2.params,
        "ee": logit_q2.bse,
        "OR": np.exp(logit_q2.params),
        "OR_ic_inf": _ic[0],
        "OR_ic_sup": _ic[1],
    }).with_columns(pl.col(pl.Float64).round(4))

    _metricas = pl.concat([
        tesina.evaluar("logistica_dif_Q2", train["victoria_local"],
                       (_prob >= 0.5).astype(int), _prob),
        tesina.evaluar("MB2_lider", train["victoria_local"], train["dif_puntos_Q2"] >= 0),
    ]).with_columns(pl.col(pl.Float64).round(5))

    _indiferencia = -logit_q2.params["Intercept"] / logit_q2.params["dif_puntos_Q2"]

    mo.vstack([
        _coef,
        _metricas,
        mo.md(f"La logística da 50 % al local cuando la diferencia es "
              f"{_indiferencia:.2f}: el local que pierde por uno al descanso "
              "todavía es favorito."),
    ])
    return


@app.cell
def _(alt, logit_q2, pl, train):
    # Proporción de victorias del local por diferencia exacta (el tamaño es la
    # cantidad de partidos) y la curva de la logística
    _por_margen = train.group_by("dif_puntos_Q2").agg(
        partidos=pl.len(), gana_local=pl.col("victoria_local").mean(),
    )
    _grilla = pl.DataFrame({"dif_puntos_Q2": list(range(
        train["dif_puntos_Q2"].min(), train["dif_puntos_Q2"].max() + 1))})
    _grilla = _grilla.with_columns(prob=pl.Series(logit_q2.predict(_grilla).to_numpy()))

    _puntos = alt.Chart(_por_margen).mark_circle(opacity=0.6).encode(
        x=alt.X("dif_puntos_Q2:Q", title="Diferencia al final del 2do cuarto (local − visita)"),
        y=alt.Y("gana_local:Q", title="Proporción de victorias del local"),
        size=alt.Size("partidos:Q", title="Partidos"),
    )
    _curva = alt.Chart(_grilla).mark_line().encode(x="dif_puntos_Q2:Q", y="prob:Q")
    (_puntos + _curva).properties(width=520, height=320)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Diferencia final

    Regresión lineal `dif_puntos_final ~ dif_puntos_Q2`. Si los residuos son
    aproximadamente normales y de varianza constante, la diferencia final
    predicha es una normal con media `a + b · dif_puntos_Q2` y desvío `sigma`,
    que es lo que hace falta para el mercado de hándicap. La varianza constante
    se ve en la tabla de tramos (`de_final`); la forma, en la asimetría y la
    curtosis de los residuos (las dos valen 0 en una normal).
    """)
    return


@app.cell
def _(smf, train):
    lineal_q2 = smf.ols("dif_puntos_final ~ dif_puntos_Q2", data=train).fit()
    return (lineal_q2,)


@app.cell
def _(lineal_q2, mo, pl, stats):
    mo.vstack([
        pl.DataFrame({
            "variable": lineal_q2.params.index,
            "coef": lineal_q2.params,
            "ee": lineal_q2.bse,
        }).with_columns(pl.col(pl.Float64).round(4)),
        pl.DataFrame([{
            "r2": lineal_q2.rsquared,
            "sigma": lineal_q2.scale ** 0.5,
            "asimetria_resid": stats.skew(lineal_q2.resid),
            "curtosis_exceso_resid": stats.kurtosis(lineal_q2.resid),
        }]).with_columns(pl.col(pl.Float64).round(4)),
    ])
    return


@app.cell
def _(alt, pl, train):
    # Diferencia final contra diferencia al 2C, con la recta de mínimos cuadrados.
    # Debajo de la línea punteada perdió el local.
    _datos = train.select("dif_puntos_Q2", "dif_puntos_final")
    _puntos = alt.Chart(_datos).mark_circle(opacity=0.25, size=20).encode(
        x=alt.X("dif_puntos_Q2:Q", title="Diferencia al final del 2do cuarto (local − visita)"),
        y=alt.Y("dif_puntos_final:Q", title="Diferencia final (local − visita)"),
    )
    _recta = _puntos.transform_regression("dif_puntos_Q2", "dif_puntos_final").mark_line()
    _cero = alt.Chart(pl.DataFrame({"y": [0]})).mark_rule(strokeDash=[4, 4]).encode(y="y:Q")
    (_puntos + _recta + _cero).properties(width=520, height=360)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. La diferencia al 2C y las estadísticas avanzadas de tiro

    De cada equipo, acumuladas hasta el final del 2do cuarto, en escala 0-100
    (TC = tiros de campo, de 2 y de 3; los mates cuentan como tiros de 2):

    | Métrica | Fórmula | Qué mide |
    |---|---|---|
    | `efg` | 100 · (T2A + 1,5·T3A) / TCI | Tiro de campo efectivo (ya está en la matriz) |
    | `ts` | 100 · puntos / (2 · (TCI + 0,44·T1I)) | Tiro verdadero: suma los libres |
    | `pct2` | 100 · T2A / T2I | Acierto en dobles |
    | `pct3` | 100 · T3A / T3I | Acierto en triples |
    | `pct1` | 100 · T1A / T1I | Acierto en libres (nulo si el equipo no tiró ninguno) |
    | `tasa3` | 100 · T3I / TCI | Qué parte de los tiros de campo son triples |
    | `tasa1` | 100 · T1I / TCI | Libres intentados por tiro de campo |
    | `asistidas` | 100 · asistencias / TCA | Qué parte de las canastas de campo fueron asistidas |
    | `tci` | TCI | Volumen: tiros de campo intentados |

    Se usan en diferencia local − visita (`dif_efg`, `dif_ts`, etc.), en
    puntos porcentuales.
    """)
    return


@app.cell
def _(pl, train):
    # Métricas de tiro de cada equipo
    tiro = train
    for _e in ["local", "visita"]:
        _tci = pl.col(f"{_e}_T2I") + pl.col(f"{_e}_T3I")
        _tca = pl.col(f"{_e}_T2A") + pl.col(f"{_e}_T3A")
        _puntos = 2 * pl.col(f"{_e}_T2A") + 3 * pl.col(f"{_e}_T3A") + pl.col(f"{_e}_T1A")
        tiro = tiro.with_columns(
            (100 * _puntos / (2 * (_tci + 0.44 * pl.col(f"{_e}_T1I")))).alias(f"{_e}_ts"),
            (100 * pl.col(f"{_e}_T2A") / pl.col(f"{_e}_T2I")).alias(f"{_e}_pct2"),
            (100 * pl.col(f"{_e}_T3A") / pl.col(f"{_e}_T3I")).alias(f"{_e}_pct3"),
            pl.when(pl.col(f"{_e}_T1I") > 0)
            .then(100 * pl.col(f"{_e}_T1A") / pl.col(f"{_e}_T1I"))
            .alias(f"{_e}_pct1"),
            (100 * pl.col(f"{_e}_T3I") / _tci).alias(f"{_e}_tasa3"),
            (100 * pl.col(f"{_e}_T1I") / _tci).alias(f"{_e}_tasa1"),
            (100 * pl.col(f"{_e}_asistencias") / _tca).alias(f"{_e}_asistidas"),
            _tci.alias(f"{_e}_tci"),
        )

    # Diferencias local − visita
    metricas = ["efg", "ts", "pct2", "pct3", "pct1", "tasa3", "tasa1", "asistidas", "tci"]
    tiro = tiro.with_columns(
        [(pl.col(f"local_{_m}") - pl.col(f"visita_{_m}")).alias(f"dif_{_m}")
         for _m in metricas],
    )
    return metricas, tiro


@app.cell
def _(metricas, mo, pl, tiro):
    # Descriptiva: media de cada equipo y desvío de la diferencia
    _desc = pl.DataFrame([
        {"metrica": _m,
         "media_local": tiro[f"local_{_m}"].mean(),
         "media_visita": tiro[f"visita_{_m}"].mean(),
         "de_dif": tiro[f"dif_{_m}"].std()}
        for _m in metricas
    ]).with_columns(pl.col(pl.Float64).round(4))

    mo.vstack([
        mo.md(f"""
    ### Descriptiva

    `pct1` (y su diferencia) es nulo en {tiro['dif_pct1'].is_null().sum()}
    partidos en que algún equipo no tiró libres en la primera mitad; esos
    partidos quedan afuera sólo en lo que usa `pct1`.
    """),
        _desc,
    ])
    return


@app.cell
def _(metricas, mo, pl, tiro):
    correlaciones = pl.DataFrame([
        {"metrica": f"dif_{_m}",
         "r_Q2": tiro.select(pl.corr(f"dif_{_m}", "dif_puntos_Q2")).item(),
         "r_final": tiro.select(pl.corr(f"dif_{_m}", "dif_puntos_final")).item(),
         "r_2T": tiro.select(pl.corr(f"dif_{_m}", "margen_2T")).item()}
        for _m in metricas
    ]).with_columns(pl.col(pl.Float64).round(4))

    mo.vstack([
        mo.md("""
    ### Correlaciones

    Correlación de cada métrica con la diferencia al 2C (`r_Q2`), con la
    diferencia final (`r_final`) y con el margen del 2do tiempo (`r_2T`).
    """),
        correlaciones,
    ])
    return (correlaciones,)


@app.cell
def _(alt, metricas, pl, tiro):
    # Diferencia al 2C contra cada métrica, con su recta de mínimos cuadrados
    _largo = (
        tiro.select("dif_puntos_Q2", *[f"dif_{_m}" for _m in metricas])
        .unpivot(index="dif_puntos_Q2", variable_name="metrica", value_name="valor")
        .drop_nulls()
        .with_columns(pl.col("valor").round(2))
    )
    _puntos = alt.Chart().mark_circle(opacity=0.15, size=10).encode(
        x=alt.X("valor:Q", title=None),
        y=alt.Y("dif_puntos_Q2:Q", title="Diferencia al 2C"),
    )
    _recta = _puntos.transform_regression("valor", "dif_puntos_Q2", groupby=["metrica"]).mark_line()
    (
        alt.layer(_puntos, _recta, data=_largo)
        .properties(width=170, height=140)
        .facet(facet=alt.Facet("metrica:N", title=None,
                               sort=[f"dif_{_m}" for _m in metricas]), columns=3)
        .resolve_scale(x="independent")
    )
    return


@app.cell
def _(correlaciones, lineal_q2, logit_q2, mo, np):
    _mas = correlaciones.sort("r_Q2", descending=True).row(0, named=True)
    mo.md(f"""
    ## Lectura

    - **Victoria.** Cada punto de diferencia al 2C multiplica por
      {np.exp(logit_q2.params['dif_puntos_Q2']):.3f} los odds de
      que gane el local.
    - **Diferencia final.** `dif_puntos_final` ≈
      {lineal_q2.params['Intercept']:.2f} + {lineal_q2.params['dif_puntos_Q2']:.3f}
      · `dif_puntos_Q2`, con desvío residual {lineal_q2.scale ** 0.5:.2f}
      puntos y R² {lineal_q2.rsquared:.3f}: la diferencia al descanso se conserva
      casi entera y el 2do tiempo agrega un ruido de unos
      {lineal_q2.scale ** 0.5:.0f} puntos, con algo de ventaja del local.
    - **Tiro.** La métrica más correlacionada con la diferencia al 2C es
      `{_mas['metrica']}` (r = {_mas['r_Q2']:.3f}); con la diferencia final
      baja a {_mas['r_final']:.3f} y con el margen del 2do tiempo es
      {_mas['r_2T']:.3f}. Lo que dice el tiro de la primera mitad sobre el
      final pasa casi todo por el marcador.

    Todo esto es sobre el train, con variables elegidas mirando esos mismos
    datos. Si algo entra a un modelo, se calcula en `etl/05_datos.R` y se
    compara en la validación.
    """)
    return


if __name__ == "__main__":
    app.run()
