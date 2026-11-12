# ==============================================================================
# tesina.py — Lo que comparten todos los notebooks de modelado
# ==============================================================================
# Cada notebook hace tres cosas:
#
#   train, validacion, test = cargar_split(2)                  # 1. cargar
#   ...ajuste del modelo, propio de cada notebook...           # 2. ajustar
#   evaluar("LC", validacion["victoria_local"], pred, prob)    # 3. evaluar
#
# y guarda sus predicciones con guardar_predicciones(). El informe
# (resultados.py) las junta con evaluar_corrida().
#
# Los modelos de la diferencia de puntos final que dan una distribución
# predictiva (los bayesianos) guardan además esa predictiva con
# guardar_predictiva(), y el informe la evalúa con evaluar_predictiva().
#
# Los modelos se comparan en la validación. El test se usa una sola vez, para
# el modelo elegido reajustado con train + validación:
# guardar_predicciones(..., conjunto="test").
# ==============================================================================

from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

# Este archivo está en modelos/codigo/: la raíz del repo está dos niveles arriba.
RAIZ = Path(__file__).resolve().parents[2]

CONJUNTOS = ["train", "validacion", "test"]


# ------------------------------------------------------------------------------
# 1. Cargar
# ------------------------------------------------------------------------------
def cargar_split(corte):
    """Train, validación y test del corte pedido, tal como los exportó
    R (etl/06_split.R), ordenados por fecha. Columnas: id_match, fecha,
    temporada, jornada, victoria_local (1 = ganó el local), dif_puntos_final,
    total_puntos_final, prorroga (1 = hubo prórroga), y después las features.
    Los NA de R quedan como null.
    """
    # Sin infer_schema_length=None, polars adivina los tipos con las primeras
    # 100 filas y toma jornada ("1", "2", ..., "Cuartos (1º)") como entero. Sin
    # null_values="NA", lee como texto, sin avisar, las columnas con faltantes.
    m = pl.read_csv(RAIZ / "datos" / "matrices" / f"matriz_{corte}c.csv",
                    try_parse_dates=True, infer_schema_length=None,
                    null_values="NA")
    return tuple(m.filter(pl.col("conjunto") == c).drop("conjunto")
                 for c in CONJUNTOS)


# ------------------------------------------------------------------------------
# 3. Evaluar
# ------------------------------------------------------------------------------
def _es_local(x):
    """Pasa 1/0, True/False o "local"/"visita" a un vector booleano."""
    x = pl.Series(x)
    if x.dtype == pl.Boolean:
        return x.to_numpy()
    if x.dtype.is_numeric():
        return (x == 1).to_numpy()
    return (x.cast(pl.String) == "local").to_numpy()


def evaluar(modelo, y_real, pred, prob_local=None):
    """Una fila con las métricas de un modelo: modelo | métricas. Es la ÚNICA
    implementación de las métricas del proyecto.

    y_real     : resultado real; 1/0, True/False o "local"/"visita".
    pred       : clase predicha, en cualquiera de esos formatos. El umbral
                 (0,5) lo aplica cada notebook, no esta función.
    prob_local : probabilidad de que gane el local. Sin ella no hay AUC, Brier
                 ni log-loss (los modelos base no tienen).
    """
    y = _es_local(y_real)
    p = _es_local(pred)

    tp = int(np.sum(p & y))
    tn = int(np.sum(~p & ~y))
    fp = int(np.sum(p & ~y))
    fn = int(np.sum(~p & y))

    def cociente(a, b):
        return a / b if b > 0 else np.nan

    sens = cociente(tp, tp + fn)   # recall de "local"
    spec = cociente(tn, tn + fp)   # recall de "visita"
    vpp = cociente(tp, tp + fp)
    vpn = cociente(tn, tn + fn)

    def f1(a, b):
        if np.isnan(a) or np.isnan(b) or a + b == 0:
            return np.nan
        return 2 * a * b / (a + b)

    auc = brier = log_loss = np.nan
    if prob_local is not None:
        q = np.asarray(prob_local, dtype=float)
        auc = roc_auc_score(y, q)
        brier = np.mean((q - y) ** 2)
        # Sin recortar las probabilidades: si un modelo dice 0 o 1 y se
        # equivoca, el log-loss da infinito, y está bien que se vea.
        with np.errstate(divide="ignore"):
            log_loss = -np.mean(np.where(y, np.log(q), np.log(1 - q)))

    return pl.DataFrame([{
        "modelo": modelo,
        "accuracy": (tp + tn) / len(y),
        "auc": auc,
        "brier": brier,
        "log_loss": log_loss,
        "sensibilidad": sens,
        "especificidad": spec,
        "vpp": vpp,
        "vpn": vpn,
        "f1_local": f1(vpp, sens),
        "f1_visita": f1(vpn, spec),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }])


def umbral_optimo(y_real, prob_local):
    """El umbral que maximiza el accuracy, en una grilla de 0,01 a 0,99; ante
    empates, el más cercano a 0,5. No se usa para clasificar (la clase sale
    siempre de 0,5): se reporta como señal de calibración, porque en un
    modelo calibrado el umbral que maximiza el accuracy está cerca de 0,5."""
    y = _es_local(y_real)
    prob = np.asarray(prob_local)
    umbrales = np.round(np.arange(0.01, 1.0, 0.01), 2)
    acc = np.array([((prob >= u) == y).mean() for u in umbrales])
    mejores = umbrales[acc == acc.max()]
    return float(mejores[np.argmin(np.abs(mejores - 0.5))])


# ------------------------------------------------------------------------------
# Guardar predicciones
# ------------------------------------------------------------------------------
def guardar_predicciones(modelo, corte, datos, pred, prob_local=None, notas="",
                         conjunto="validacion"):
    """Deja predicciones/<corte>c/<conjunto>/pred_<modelo>.csv, una fila por
    partido (id_match, fecha, y_real, pred, prob_local). Es lo único que se
    guarda de cada modelo.

    datos    : los partidos predichos: exactamente los del conjunto, en orden.
    notas    : qué es el modelo y qué variables usa; va a
               pred_<modelo>.notas.txt y el informe la muestra.
    conjunto : "validacion" o "test" (sólo el modelo elegido, una vez).
    """
    _, validacion, test = cargar_split(corte)
    esperado = {"validacion": validacion, "test": test}[conjunto]
    if list(datos["id_match"]) != list(esperado["id_match"]):
        raise ValueError(f"'datos' no son los {len(esperado)} partidos de "
                         f"{conjunto} del corte {corte}C, en el mismo orden.")

    carpeta = RAIZ / "predicciones" / f"{corte}c" / conjunto
    carpeta.mkdir(parents=True, exist_ok=True)
    archivo = carpeta / f"pred_{modelo}.csv"

    a_texto = lambda v: np.where(_es_local(v), "local", "visita")  # noqa: E731
    pl.DataFrame({
        "id_match": datos["id_match"],
        "fecha": datos["fecha"].dt.to_string("%Y-%m-%d"),
        "y_real": a_texto(datos["victoria_local"]),
        "pred": a_texto(pred),
        "prob_local": (np.asarray(prob_local, dtype=float)
                       if prob_local is not None else None),
    }).write_csv(archivo)

    archivo_notas = carpeta / f"pred_{modelo}.notas.txt"
    if notas:
        archivo_notas.write_text(notas, encoding="utf-8")
    elif archivo_notas.exists():
        archivo_notas.unlink()   # no dejar la nota de una versión anterior


# ------------------------------------------------------------------------------
# Evaluar todas las predicciones guardadas de un corte
# ------------------------------------------------------------------------------
def evaluar_corrida(corte, conjunto="validacion"):
    """Evalúa todos los predicciones/<corte>c/<conjunto>/pred_*.csv contra ese
    conjunto. Devuelve dos tablas: `resultados` (una fila por modelo, con las
    métricas de evaluar() y la nota del modelo) y `salteados` (los archivos
    que no traen exactamente los partidos del conjunto: sus métricas no serían
    comparables con las del resto).
    """
    _, validacion, test = cargar_split(corte)
    verdad = ({"validacion": validacion, "test": test}[conjunto]
              .select("id_match", "victoria_local"))

    filas, salteados = [], []
    for archivo in sorted((RAIZ / "predicciones" / f"{corte}c" / conjunto).glob("pred_*.csv")):
        modelo = archivo.stem.removeprefix("pred_")
        p = pl.read_csv(archivo)

        if (len(p) != len(verdad) or p["id_match"].is_duplicated().any()
                or set(p["id_match"]) != set(verdad["id_match"])):
            salteados.append((modelo, f"sus {len(p)} partidos no son los "
                                      f"{len(verdad)} de {conjunto}"))
            continue
        j = verdad.join(p, on="id_match", maintain_order="left")

        prob = None if j["prob_local"].is_null().all() else j["prob_local"]
        archivo_notas = archivo.with_suffix(".notas.txt")
        notas = (archivo_notas.read_text(encoding="utf-8").strip()
                 if archivo_notas.exists() else "")
        filas.append(evaluar(modelo, j["victoria_local"], j["pred"], prob)
                     .with_columns(notas=pl.lit(notas)))

    resultados = pl.concat(filas) if filas else pl.DataFrame()
    return resultados, pl.DataFrame(salteados, schema=["modelo", "motivo"],
                                    orient="row")


# ------------------------------------------------------------------------------
# Diferencia de puntos final: guardar y evaluar la distribución predictiva
# ------------------------------------------------------------------------------
def guardar_predictiva(modelo, corte, datos, y_sim, log_lik, conjunto="validacion"):
    """Deja predicciones/<corte>c/<conjunto>/sim_<modelo>.csv, en formato
    largo: una fila por partido y simulación (id_match, sim, y_sim, log_lik).
    Es aparte de guardar_predicciones(), que guarda la probabilidad de
    victoria del mismo modelo, con sus notas.

    datos   : los partidos predichos: exactamente los del conjunto, en orden.
    y_sim   : matriz partidos × simulaciones: en cada columna, una diferencia
              final simulada de la predictiva, con una muestra de la posterior.
    log_lik : matriz partidos × simulaciones: el log de la densidad de la
              diferencia final observada, con la misma muestra de la posterior
              que la columna de y_sim.
    """
    _, validacion, test = cargar_split(corte)
    esperado = {"validacion": validacion, "test": test}[conjunto]
    if list(datos["id_match"]) != list(esperado["id_match"]):
        raise ValueError(f"'datos' no son los {len(esperado)} partidos de "
                         f"{conjunto} del corte {corte}C, en el mismo orden.")

    carpeta = RAIZ / "predicciones" / f"{corte}c" / conjunto
    carpeta.mkdir(parents=True, exist_ok=True)
    n_partidos, n_sim = np.shape(y_sim)
    pl.DataFrame({
        "id_match": np.repeat(datos["id_match"].to_numpy(), n_sim),
        "sim": np.tile(np.arange(1, n_sim + 1), n_partidos),
        "y_sim": np.ravel(y_sim),
        "log_lik": np.ravel(log_lik),
    }).write_csv(carpeta / f"sim_{modelo}.csv")


def evaluar_predictiva(corte, conjunto="validacion"):
    """Evalúa la distribución predictiva de la diferencia final de todos los
    predicciones/<corte>c/<conjunto>/sim_*.csv contra ese conjunto. Es la
    ÚNICA implementación de estas métricas. Devuelve tres tablas:

    resultados : una fila por modelo.
                 log_dens_pred: promedio, sobre los partidos, del log de la
                 densidad predictiva en la diferencia final observada,
                 log((1/S) Σ_s p(y | θ_s)). Es el log-loss de una respuesta
                 continua con el signo cambiado: más alto es mejor.
                 cobertura_50 y cobertura_90: proporción de partidos cuya
                 diferencia final cae en el intervalo predictivo central del
                 50 % y del 90 %.
    pit        : una fila por modelo y partido: la proporción de simulaciones
                 por debajo de la diferencia observada. Si la predictiva está
                 calibrada, es uniforme entre 0 y 1.
    salteados  : los archivos que no traen exactamente los partidos del
                 conjunto.
    """
    _, validacion, test = cargar_split(corte)
    verdad = ({"validacion": validacion, "test": test}[conjunto]
              .select("id_match", pl.col("dif_puntos_final").alias("y")))

    filas, pits, salteados = [], [], []
    for archivo in sorted((RAIZ / "predicciones" / f"{corte}c" / conjunto).glob("sim_*.csv")):
        modelo = archivo.stem.removeprefix("sim_")
        s = pl.read_csv(archivo)

        if set(s["id_match"]) != set(verdad["id_match"]):
            salteados.append((modelo, f"sus {s['id_match'].n_unique()} partidos no son "
                                      f"los {len(verdad)} de {conjunto}"))
            continue

        # Por partido. El log de la densidad promedio se calcula restando el
        # máximo antes de exponenciar, para no perder precisión.
        por_partido = (
            s.join(verdad, on="id_match")
            .group_by("id_match")
            .agg(
                lpd=(pl.col("log_lik") - pl.col("log_lik").max()).exp().mean().log()
                    + pl.col("log_lik").max(),
                pit=(pl.col("y_sim") < pl.col("y")).mean(),
                adentro_50=pl.col("y").first().is_between(pl.col("y_sim").quantile(0.25),
                                                          pl.col("y_sim").quantile(0.75)),
                adentro_90=pl.col("y").first().is_between(pl.col("y_sim").quantile(0.05),
                                                          pl.col("y_sim").quantile(0.95)),
            )
        )
        filas.append({"modelo": modelo,
                      "log_dens_pred": por_partido["lpd"].mean(),
                      "cobertura_50": por_partido["adentro_50"].mean(),
                      "cobertura_90": por_partido["adentro_90"].mean()})
        pits.append(por_partido.select(pl.lit(modelo).alias("modelo"), "id_match", "pit"))

    return (pl.DataFrame(filas) if filas else pl.DataFrame(),
            pl.concat(pits).sort("modelo", "id_match") if pits else pl.DataFrame(),
            pl.DataFrame(salteados, schema=["modelo", "motivo"], orient="row"))
