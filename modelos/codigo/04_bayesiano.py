import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import altair as alt
    # arviz_stats es la parte de cálculo de ArviZ (viene con PyMC); el paquete
    # arviz entero carga también la de gráficos, que pide matplotlib
    import arviz_stats as azs
    import marimo as mo
    import numpy as np
    import polars as pl
    import pymc as pm
    from scipy import stats

    import tesina

    return alt, azs, mo, np, pl, pm, stats, tesina


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Modelo bayesiano de la diferencia de puntos final

    Una regresión con error **t de Student** para la diferencia final, local −
    visita, **con la prórroga incluida** (`dif_puntos_final`), con la
    información al final del 2do cuarto. Se ajusta con PyMC sobre el train.

    $$
    \texttt{dif\_puntos\_final} \sim t_\nu(\mu,\ \sigma), \qquad
    \mu = \beta_0 + \beta_1 x_1 + \dots + \beta_9 x_9
    $$

    Las variables son **las mismas de `02_logistica.py`**, armadas de la
    misma forma; lo que cambia es la respuesta. La logística mira quién ganó;
    este modelo, por cuánto, y de su distribución predictiva salen la
    probabilidad de que gane el local, P(diferencia final > 0), y la de que el
    local cubra cualquier línea de hándicap, P(diferencia final > h).

    | Grupo | Variables |
    |---|---|
    | Marcador al corte | `dif_puntos_Q2`, `total_puntos_Q2_c`, `gano_Q1` |
    | Contexto | `es_playoff`, `partidos_jugados_prom` |
    | Cuatro factores, sin eFG% | `dif_tov`, `dif_orb`, `dif_ftr` |
    | Fuerza previa | `dif_fuerza_margen` |

    **Cómo se leen los parámetros.** Cada β está en puntos de diferencia
    final por unidad de su variable, a igualdad del resto y con el marcador al
    descanso dado. β₀ es la diferencia final esperada con todas las variables
    en 0: un partido de la jornada 1, de fase regular, empatado al descanso y
    en el 1er cuarto, con ritmo promedio y equipos de igual fuerza. σ es la
    escala de lo que el modelo no explica (el resto del partido) y ν, el peso
    de las colas: con ν grande la t es casi una normal; con ν chico hay más
    palizas y remontadas que las que espera una normal. ν lo estima el modelo.

    **Estandarización.** Para muestrear, las variables continuas se
    estandarizan con la media y el desvío del train; las indicadoras
    (`gano_Q1`, `es_playoff`) quedan como están. El muestreo anda más rápido
    y las a prioris se ponen en una escala común. Después, cada muestra de la
    posterior se vuelve a las unidades originales, y con eso se informan los
    coeficientes y se predice: la estandarización no cambia el modelo ni sus
    predicciones.

    **A prioris**, poco informativas en la escala de los puntos (con las
    continuas estandarizadas, el β de cada una es por desvío de la variable):

    - β₀ ~ N(0, 5) y cada β ~ N(0, 5).
    - σ ~ media normal(15).
    - ν ~ Gamma(2; 0,1), con media 20 (Juárez y Steel, 2010).

    Abajo se chequean con la predictiva a priori: las diferencias finales que
    generan tienen que ser plausibles.

    **Chequeos predictivos a posteriori**, sobre el train: si el modelo
    ajustado reproduce rasgos de las diferencias con las que se ajustó.

    **Evaluación en la validación** (la temporada 2024-25 entera). La
    probabilidad de victoria, P(diferencia final > 0) promediada sobre la
    posterior, se guarda con `guardar_predicciones()`, con la clase en 0,5:
    entra a la tabla de `resultados.py` y al gráfico de calibración, igual que
    la logística. La distribución predictiva (2.000 simulaciones por partido y
    la log verosimilitud de lo observado) se guarda con `guardar_predictiva()`
    y `resultados.py` la evalúa con `evaluar_predictiva()`: log densidad
    predictiva, cobertura de los intervalos predictivos del 50 % y del 90 %,
    y el PIT. El **test**, la temporada 2025-26, no se usa acá.

    Las predicciones quedan en `predicciones/2c/validacion/pred_t_lineal.csv`
    y la predictiva en `predicciones/2c/validacion/sim_t_lineal.csv`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Corte 2C: datos al final del 2do cuarto
    """)
    return


@app.cell(hide_code=True)
def _(pl, tesina):
    # 1. Cargar, con las variables armadas como en 02_logistica.py. El ritmo se
    # centra con la media del train, también en la validación. Lo que es de
    # cada equipo, en diferencia local − visita; los partidos jugados, en
    # promedio. Del 1er cuarto, quién lo ganó (1, −1 o 0).
    _train, _valid, _ = tesina.cargar_split(2)
    media_ritmo_2c = _train["total_puntos_Q2"].mean()
    _nuevas = [
        (pl.col("total_puntos_Q2") - media_ritmo_2c).alias("total_puntos_Q2_c"),
        *[(pl.col(f"local_{_f}") - pl.col(f"visita_{_f}")).alias(f"dif_{_f}")
          for _f in ["tov", "orb", "ftr"]],
        (pl.col("fuerza_margen_local") - pl.col("fuerza_margen_visita")).alias("dif_fuerza_margen"),
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
def _(np, train_2c, variables_2c):
    # Estandarizar las continuas con la media y el desvío del train. Las
    # indicadoras quedan como están (media 0 y desvío 1 en la cuenta, para que
    # la vuelta a las unidades originales sea la misma). Las predicciones se
    # hacen con los coeficientes vueltos a las unidades originales, así que la
    # validación no se estandariza.
    _continua = ~np.isin(variables_2c, ["gano_Q1", "es_playoff"])
    _x = train_2c.select(variables_2c).to_numpy().astype(float)
    media_x_2c = np.where(_continua, _x.mean(axis=0), 0.0)
    desvio_x_2c = np.where(_continua, _x.std(axis=0, ddof=1), 1.0)
    x_train_2c = (_x - media_x_2c) / desvio_x_2c
    y_train_2c = train_2c["dif_puntos_final"].to_numpy().astype(float)
    return desvio_x_2c, media_x_2c, x_train_2c, y_train_2c


@app.cell(hide_code=True)
def _(pm, variables_2c, x_train_2c, y_train_2c):
    # 2. Ajustar sobre el train. Las cuatro cadenas, en serie: correrlas en
    # paralelo desde un notebook puede trabarse en Windows, y este modelo tarda
    # segundos igual.
    with pm.Model(coords={"covariable": variables_2c}):
        _b0 = pm.Normal("b0", mu=0, sigma=5)
        _b = pm.Normal("b", mu=0, sigma=5, dims="covariable")
        _sigma = pm.HalfNormal("sigma", sigma=15)
        _nu = pm.Gamma("nu", alpha=2, beta=0.1)
        pm.StudentT("dif_puntos_final", nu=_nu, mu=_b0 + x_train_2c @ _b, sigma=_sigma,
                    observed=y_train_2c)
        a_priori_2c = pm.sample_prior_predictive(draws=1000, random_seed=2026)
        posterior_2c = pm.sample(draws=2000, tune=1000, chains=4, cores=1,
                                 random_seed=2026, progressbar=False)
    return a_priori_2c, posterior_2c


@app.cell(hide_code=True)
def _(a_priori_2c, mo, np, pl, y_train_2c):
    # Predictiva a priori: las diferencias finales que generan las a prioris,
    # con las variables del train, contra las observadas
    _y_prior = a_priori_2c.prior_predictive["dif_puntos_final"].values.ravel()
    _filas = []
    for _nombre, _y in [("a priori", _y_prior), ("observadas", y_train_2c)]:
        _q = np.quantile(_y, [0.05, 0.25, 0.5, 0.75, 0.95])
        _filas.append({"diferencias": _nombre, "q05": _q[0], "q25": _q[1], "mediana": _q[2],
                       "q75": _q[3], "q95": _q[4], "prop_mas_de_30": np.mean(np.abs(_y) > 30),
                       "prop_mas_de_60": np.mean(np.abs(_y) > 60)})

    mo.vstack([
        mo.md(f"""
    ### Predictiva a priori

    Diferencias finales simuladas con las a prioris (1.000 simulaciones de los
    {len(y_train_2c)} partidos del train) contra las observadas. Las a prioris
    tienen que dejar lugar a todo lo que se observa sin concentrarse en
    valores imposibles; la diferencia observada más grande, en valor absoluto,
    es {np.abs(y_train_2c).max():.0f}.
    """),
        pl.DataFrame(_filas).with_columns(pl.col(pl.Float64).round(3)),
    ])
    return


@app.cell(hide_code=True)
def _(a_priori_2c, alt, mo, np, pl, stats, y_train_2c):
    # Densidad de las diferencias observadas contra la de 50 réplicas de la
    # predictiva a priori (el ppc_dens_overlay de bayesplot). Las densidades
    # se evalúan en 81 puntos y se redondean: los datos de cada gráfico van
    # enteros al HTML.
    _rep = a_priori_2c.prior_predictive["dif_puntos_final"].values[0, :50]   # réplicas × partidos
    _grilla = np.linspace(-120, 120, 81)
    _replicas = pl.concat([
        pl.DataFrame({"diferencia": _grilla, "densidad": stats.gaussian_kde(_r)(_grilla).round(5),
                      "replica": _i})
        for _i, _r in enumerate(_rep)
    ])
    _observadas = pl.DataFrame({"diferencia": _grilla,
                                "densidad": stats.gaussian_kde(y_train_2c)(_grilla).round(5)})
    _x_eje = alt.X("diferencia:Q", title="Diferencia final, local − visita")
    _lineas_rep = alt.Chart(_replicas).mark_line(strokeWidth=1, opacity=0.35, color="#f58518").encode(
        x=_x_eje, y=alt.Y("densidad:Q", title="Densidad"), detail="replica:N")
    _linea_obs = alt.Chart(_observadas).mark_line(strokeWidth=3, color="#4c78a8").encode(
        x=_x_eje, y="densidad:Q")

    mo.vstack([
        mo.md(f"""
    La misma comparación en un gráfico: la densidad de las
    {len(y_train_2c)} diferencias finales del train (en azul) contra la de 50
    réplicas de la predictiva a priori (en naranja), cada una con los mismos
    partidos. Las réplicas tienen que cubrir lo observado y pueden extenderse
    más allá, pero sin poner mucha probabilidad en diferencias que no ocurren.
    """),
        (_lineas_rep + _linea_obs).properties(width=620, height=260),
    ])
    return


@app.cell(hide_code=True)
def _(desvio_x_2c, media_x_2c, posterior_2c):
    # Las muestras de la posterior (4 cadenas × 2.000), con los coeficientes
    # vueltos a las unidades originales de cada variable
    _p = posterior_2c.posterior
    _b0_est = _p["b0"].values.ravel()
    _b_est = _p["b"].values.reshape(len(_b0_est), -1)
    b_2c = _b_est / desvio_x_2c
    b0_2c = _b0_est - (_b_est * media_x_2c / desvio_x_2c).sum(axis=1)
    sigma_2c = _p["sigma"].values.ravel()
    nu_2c = _p["nu"].values.ravel()
    return b0_2c, b_2c, nu_2c, sigma_2c


@app.cell(hide_code=True)
def _(azs, b0_2c, b_2c, mo, np, nu_2c, pl, posterior_2c, sigma_2c, train_2c, variables_2c):
    # Convergencia y coeficientes. ArviZ devuelve el resumen en pandas: se pasa
    # a polars apenas sale
    _diag = pl.from_pandas(azs.summary(posterior_2c, var_names=["b0", "b", "sigma", "nu"]))
    _divergencias = int(posterior_2c.sample_stats["diverging"].sum())
    _muestras = np.column_stack([b0_2c, b_2c, sigma_2c, nu_2c])
    _nombres = ["intercepto", *variables_2c, "sigma", "nu"]

    mo.vstack([
        mo.md(f"""
    ### Posterior

    Ajustado sobre los {len(train_2c)} partidos del train: 4 cadenas de 2.000
    muestras, después de 1.000 de calentamiento. R-hat máximo
    {_diag["r_hat"].max():.3f} (tiene que estar cerca de 1),
    tamaño efectivo de muestra mínimo {_diag["ess_bulk"].min():.0f}
    y {_divergencias} transiciones divergentes.

    Coeficientes en las unidades originales: media y desvío de la posterior,
    e intervalo de credibilidad del 95 % (percentiles 2,5 y 97,5).
    """),
        pl.DataFrame({
            "parametro": _nombres,
            "media": _muestras.mean(axis=0),
            "desvio": _muestras.std(axis=0, ddof=1),
            "ic_inf": np.quantile(_muestras, 0.025, axis=0),
            "ic_sup": np.quantile(_muestras, 0.975, axis=0),
        }).with_columns(pl.col(pl.Float64).round(4)),
    ])
    return


@app.cell(hide_code=True)
def _(alt, b0_2c, b_2c, mo, np, nu_2c, pl, sigma_2c, train_2c, variables_2c, y_train_2c):
    # Chequeos predictivos a posteriori, en el train: con cada muestra de la
    # posterior, una réplica de las diferencias finales del train. Las réplicas
    # se redondean, porque las diferencias reales son enteras.
    _x = train_2c.select(variables_2c).to_numpy().astype(float)
    _mu = b0_2c + _x @ b_2c.T                      # partidos × muestras
    _rng = np.random.default_rng(2026)
    y_rep_2c = np.round(_mu + sigma_2c * _rng.standard_t(nu_2c, size=_mu.shape))

    # Cada estadístico: el observado y el de cada réplica
    _estadisticos = [
        ("media", y_train_2c.mean(), y_rep_2c.mean(axis=0)),
        ("desvío", y_train_2c.std(ddof=1), y_rep_2c.std(axis=0, ddof=1)),
        ("prop. |dif| ≤ 3", np.mean(np.abs(y_train_2c) <= 3), (np.abs(y_rep_2c) <= 3).mean(axis=0)),
        ("prop. |dif| ≥ 20", np.mean(np.abs(y_train_2c) >= 20), (np.abs(y_rep_2c) >= 20).mean(axis=0)),
    ]
    _filas = []
    for _nombre, _obs, _rep in _estadisticos:
        _filas.append({"estadistico": _nombre, "observado": _obs,
                       "replicas_media": _rep.mean(),
                       "replicas_q05": np.quantile(_rep, 0.05),
                       "replicas_q95": np.quantile(_rep, 0.95),
                       "p_valor": np.mean(_rep >= _obs)})

    # Histograma de a 3 puntos, centrado en el 0: las barras son lo observado;
    # los puntos, la media de las réplicas, con el intervalo del 90 %
    _bordes = np.arange(-49.5, 50.5, 3)
    _centros = (_bordes[:-1] + _bordes[1:]) / 2
    _rep_hist = np.column_stack([np.histogram(y_rep_2c[:, _s], _bordes)[0]
                                 for _s in range(y_rep_2c.shape[1])])
    _observadas = pl.DataFrame({"centro": _centros,
                                "partidos": np.histogram(y_train_2c, _bordes)[0],
                                "serie": "Observadas"})
    _replicas = pl.DataFrame({"centro": _centros,
                              "media": _rep_hist.mean(axis=1),
                              "q05": np.quantile(_rep_hist, 0.05, axis=1),
                              "q95": np.quantile(_rep_hist, 0.95, axis=1),
                              "serie": "Réplicas: media e intervalo del 90 %"})
    _color = alt.Color("serie:N", title=None,
                       scale=alt.Scale(domain=["Observadas", "Réplicas: media e intervalo del 90 %"],
                                       range=["#4c78a8", "#f58518"]),
                       legend=alt.Legend(orient="top", labelLimit=400))
    _x_eje = alt.X("centro:Q", title="Diferencia final, local − visita (de a 3 puntos)")
    _barras = alt.Chart(_observadas).mark_bar(width=12).encode(
        x=_x_eje, y=alt.Y("partidos:Q", title="Partidos"), color=_color,
        tooltip=[alt.Tooltip("centro:Q", title="Centro"), alt.Tooltip("partidos:Q", title="Partidos")])
    _rangos = alt.Chart(_replicas).mark_rule(strokeWidth=2).encode(
        x=_x_eje, y="q05:Q", y2="q95:Q", color=_color)
    _medias = alt.Chart(_replicas).mark_point(filled=True, size=40, opacity=1).encode(
        x=_x_eje, y="media:Q", color=_color,
        tooltip=[alt.Tooltip("centro:Q", title="Centro"),
                 alt.Tooltip("media:Q", title="Media de las réplicas", format=".1f"),
                 alt.Tooltip("q05:Q", title="q05", format=".0f"),
                 alt.Tooltip("q95:Q", title="q95", format=".0f")])

    mo.vstack([
        mo.md(f"""
    ### Chequeos predictivos a posteriori (train)

    Con cada una de las {len(b0_2c)} muestras de la posterior se simula una
    réplica de las {len(y_train_2c)} diferencias finales del train, y se
    compara con lo observado. El p-valor es la proporción de réplicas con el
    estadístico mayor o igual que el observado: cerca de 0 o de 1, el modelo
    no reproduce ese rasgo de los datos. Como el chequeo usa los mismos datos
    con los que se ajustó, es indulgente: si algo falla acá, el problema es
    real.

    La prórroga hace que ningún partido termine empatado y que haya menos
    diferencias chicas que las que espera una distribución continua: es lo
    que miran la proporción de |dif| ≤ 3 y la barra del centro del histograma.
    """),
        pl.DataFrame(_filas).with_columns(pl.col(pl.Float64).round(4)),
        (_barras + _rangos + _medias).properties(width=620, height=260),
    ])
    return (y_rep_2c,)


@app.cell(hide_code=True)
def _(alt, mo, np, pl, stats, y_rep_2c, y_train_2c):
    # Densidad de las diferencias observadas contra la de 50 réplicas de la
    # predictiva a posteriori (el ppc_dens_overlay de bayesplot), igual que la
    # de la predictiva a priori
    _grilla = np.linspace(-60, 60, 81)
    _replicas = pl.concat([
        pl.DataFrame({"diferencia": _grilla,
                      "densidad": stats.gaussian_kde(y_rep_2c[:, _s])(_grilla).round(5),
                      "replica": _s})
        for _s in range(50)
    ])
    _observadas = pl.DataFrame({"diferencia": _grilla,
                                "densidad": stats.gaussian_kde(y_train_2c)(_grilla).round(5)})
    _x_eje = alt.X("diferencia:Q", title="Diferencia final, local − visita")
    _lineas_rep = alt.Chart(_replicas).mark_line(strokeWidth=1, opacity=0.35, color="#f58518").encode(
        x=_x_eje, y=alt.Y("densidad:Q", title="Densidad"), detail="replica:N")
    _linea_obs = alt.Chart(_observadas).mark_line(strokeWidth=3, color="#4c78a8").encode(
        x=_x_eje, y="densidad:Q")

    mo.vstack([
        mo.md("""
    La densidad de las diferencias finales del train (en azul) contra la de
    50 de las réplicas (en naranja). Se ve lo mismo que en el histograma: las
    réplicas tienen un solo pico y la observada tiene un pozo en el 0, por la
    prórroga.
    """),
        (_lineas_rep + _linea_obs).properties(width=620, height=260),
    ])
    return


@app.cell(hide_code=True)
def _(alt, mo, np, pl, train_2c, y_rep_2c, y_train_2c):
    # Intervalos predictivos partido a partido contra la diferencia al
    # descanso (el ppc_intervals de bayesplot): los cuantiles de las réplicas
    # de cada partido
    _q = np.quantile(y_rep_2c, [0.05, 0.25, 0.75, 0.95], axis=1)
    _intervalos = pl.DataFrame({
        "dif_puntos_Q2": train_2c["dif_puntos_Q2"], "observada": y_train_2c,
        "q05": _q[0], "q25": _q[1], "q75": _q[2], "q95": _q[3],
    })
    _adentro_50 = np.mean((y_train_2c >= _q[1]) & (y_train_2c <= _q[2]))
    _adentro_90 = np.mean((y_train_2c >= _q[0]) & (y_train_2c <= _q[3]))

    _x_eje = alt.X("dif_puntos_Q2:Q", title="Diferencia al final del 2do cuarto (local − visita)")
    _int_90 = alt.Chart(_intervalos).mark_rule(strokeWidth=1, opacity=0.3, color="#f58518").encode(
        x=_x_eje, y=alt.Y("q05:Q", title="Diferencia final, local − visita"), y2="q95:Q")
    _int_50 = alt.Chart(_intervalos).mark_rule(strokeWidth=3, opacity=0.5, color="#f58518").encode(
        x=_x_eje, y="q25:Q", y2="q75:Q")
    _puntos = alt.Chart(_intervalos).mark_circle(size=12, opacity=0.8, color="#4c78a8").encode(
        x=_x_eje, y="observada:Q")

    mo.vstack([
        mo.md(f"""
    Intervalos predictivos partido a partido: para cada partido del train, el
    intervalo central del 50 % (línea gruesa) y el del 90 % (línea fina) de
    sus réplicas, y la diferencia final observada (punto), contra la
    diferencia al descanso. Los partidos con la misma diferencia al descanso
    tienen intervalos distintos porque cambian las otras variables. Si el
    modelo capta cómo cambia la diferencia final con `dif_puntos_Q2`, los
    puntos quedan adentro en la misma proporción en todo el rango: en el
    train, {_adentro_50:.3f} adentro del intervalo del 50 % y {_adentro_90:.3f}
    adentro del de 90 %.
    """),
        (_int_90 + _int_50 + _puntos).properties(width=620, height=360),
    ])
    return


@app.cell(hide_code=True)
def _(
    b0_2c,
    b_2c,
    mo,
    np,
    nu_2c,
    pl,
    sigma_2c,
    stats,
    tesina,
    valid_2c,
    variables_2c,
):
    # 3. Predecir la validación con cada muestra de la posterior.
    # P(gana el local) = P(dif_puntos_final > 0), promediada sobre la
    # posterior; la clase sale de un umbral de 0,5.
    _x = valid_2c.select(variables_2c).to_numpy().astype(float)
    _y = valid_2c["dif_puntos_final"].to_numpy().astype(float)
    _mu = b0_2c + _x @ b_2c.T                      # partidos × muestras
    prob_2c = stats.t.sf(0, df=nu_2c, loc=_mu, scale=sigma_2c).mean(axis=1)
    pred_2c = (prob_2c >= 0.5).astype(int)
    umbral_senal_2c = tesina.umbral_optimo(valid_2c["victoria_local"], prob_2c)

    # La predictiva que se guarda: con una de cada cuatro muestras de la
    # posterior (2.000), una diferencia final simulada por partido y el log de
    # la densidad de la diferencia observada
    _cada4 = slice(None, None, 4)
    _mu4, _sigma4, _nu4 = _mu[:, _cada4], sigma_2c[_cada4], nu_2c[_cada4]
    _rng = np.random.default_rng(2026)
    y_sim_2c = _mu4 + _sigma4 * _rng.standard_t(_nu4, size=_mu4.shape)
    log_lik_2c = stats.t.logpdf(_y[:, None], df=_nu4, loc=_mu4, scale=_sigma4)

    mo.vstack([
        mo.md(f"### Evaluación en la validación: {len(valid_2c)} partidos, del "
              f"{valid_2c['fecha'].min():%Y-%m-%d} al {valid_2c['fecha'].max():%Y-%m-%d}"),
        mo.md(f"Probabilidad de victoria del local: P(diferencia final > 0). Clase predicha "
              f"con umbral 0,5. El umbral que maximizaría el accuracy en la validación es "
              f"{umbral_senal_2c:.2f}: en un modelo calibrado, cerca de 0,5."),
        tesina.evaluar("t_lineal", valid_2c["victoria_local"], pred_2c, prob_2c)
        .select("modelo", "log_loss", "brier", pl.exclude("modelo", "log_loss", "brier"))
        .with_columns(pl.col(pl.Float64).round(5)),
    ])
    return log_lik_2c, pred_2c, prob_2c, umbral_senal_2c, y_sim_2c


@app.cell(hide_code=True)
def _(
    log_lik_2c,
    media_ritmo_2c,
    mo,
    pl,
    pm,
    pred_2c,
    prob_2c,
    tesina,
    umbral_senal_2c,
    valid_2c,
    variables_2c,
    y_sim_2c,
):
    # Guardar las predicciones y la predictiva, y evaluar la distribución
    tesina.guardar_predicciones(
        "t_lineal", 2, valid_2c, pred_2c, prob_2c,
        notas=f"Regresión bayesiana (PyMC {pm.__version__}) de dif_puntos_final, con la prórroga "
              "incluida, con error t de Student y las mismas variables que logistica: "
              f"{', '.join(variables_2c)}. Las dif_ son local − visita; dif_fuerza_margen sale de la "
              "fuerza previa por margen medio de etl/05_datos.R (los últimos 34 partidos de fase "
              "regular antes del partido, contando los de la temporada anterior). "
              "partidos_jugados_prom es el promedio de partidos_jugados_local y "
              "partidos_jugados_visita. gano_Q1 es 1 si el local ganó el 1er cuarto, −1 si lo ganó "
              "la visita y 0 si empataron. total_puntos_Q2_c es total_puntos_Q2 centrado en la "
              f"media del train ({media_ritmo_2c:.1f}). Para muestrear, las continuas se "
              "estandarizan con la media y el desvío del train. A prioris: intercepto y "
              "coeficientes N(0, 5) (los de las continuas, por desvío), sigma media normal(15), "
              "nu Gamma(2; 0,1). 4 cadenas de 2.000 muestras después de 1.000 de calentamiento, "
              "semilla 2026. prob_local = P(dif_puntos_final > 0), promediada sobre la posterior. "
              f"Umbral 0,5; el que maximizaría el accuracy en la validación es "
              f"{umbral_senal_2c:.2f}. La predictiva (2.000 simulaciones por partido) está en "
              "sim_t_lineal.csv.",
    )
    tesina.guardar_predictiva("t_lineal", 2, valid_2c, y_sim_2c, log_lik_2c)
    _resultados, _, _ = tesina.evaluar_predictiva(2)

    mo.vstack([
        mo.md(f"""
    ### La distribución predictiva en la validación: {len(valid_2c)} partidos

    `log_dens_pred` es el promedio del log de la densidad que la predictiva le
    da a la diferencia final observada: el log-loss de una respuesta continua,
    con el signo cambiado (más alto es mejor). Sirve para comparar modelos de
    la diferencia de puntos, no contra la logística. Con una predictiva
    calibrada, la cobertura de los intervalos del 50 % y del 90 % da cerca de
    0,5 y 0,9. El PIT y la comparación con otros modelos están en
    `resultados.py`.
    """),
        _resultados.filter(pl.col("modelo") == "t_lineal")
        .with_columns(pl.col(pl.Float64).round(5)),
    ])
    return


if __name__ == "__main__":
    app.run()
