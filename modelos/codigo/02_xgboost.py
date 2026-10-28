"""
02_xgboost.py — Modelo de árboles con tuneo automático (FLAML) sobre el MISMO
split y las MISMAS features que el resto del proyecto.

Reemplaza a R/02_xgboost.R, que tuneaba con una grilla fija de 20 puntos en
tidymodels. El motivo del cambio es el tuneo: FLAML busca por presupuesto de
tiempo en vez de por grilla, y no existe en R.

Qué se mantiene igual que en la versión de R, y por qué:

1. NO arma features ni split. Lee data/derived/<corte>c/, que exporta
   R/_exportar_split.R. La matriz es la misma que consume TabPFN (regla 2).
2. Validación interna con ROLLING ORIGIN EXPANDIDO, replicando exactamente los
   folds de rsample::rolling_origin(initial=600, assess=100, skip=100,
   cumulative=TRUE). El default de FLAML es CV aleatorio, que mete futuro en el
   entrenamiento: es justo el error que documenta el §6.2 del contexto.
3. Se tunea por LOG LOSS, no por accuracy. Para la parte de apuestas importa la
   calibración de la probabilidad, no sólo si la clase predicha es la correcta.
4. Clasificación sobre victoria_local, para comparar con TabPFN en igualdad de
   condiciones. La variante de regresión sobre dif_puntos_final va aparte, con
   --regresion, y se registra como OTRO modelo.

Qué NO hace, a diferencia de la versión de R: no calcula métricas. Guarda las
predicciones y las evalúa R con evaluar() (regla 3). Si este script calculara su
propio accuracy habría dos implementaciones y nada avisaría el día que dejaran
de coincidir.

Uso, desde la raíz del proyecto:

    Rscript R/_exportar_split.R 3
    uv run python py/02_xgboost.py --corte 3
    uv run python py/02_xgboost.py --corte 3 --presupuesto 900 --regresion
    Rscript R/_evaluar_predicciones.R 3
    Rscript R/04_tabla_resultados.R 3

Sin --corte corre el 2C, que es el default del proyecto.

ABLACIÓN. Con --variante <escalon> lee data/derived/<corte>c-<escalon>/ en vez
de la matriz completa, y guarda todo como FLAML_<escalon>. Los escalones los
define ESCALERA_ABLACION en R/00_datos.R y los exporta R/_exportar_ablacion.R:

    Rscript R/_exportar_ablacion.R 3
    uv run python py/02_xgboost.py --corte 3 --variante e1_marcador
    Rscript R/05_ablacion.R 3

Las filas de la ablación NO van a resultados.csv: van a runs/<corte>c/ablacion.csv,
porque varían la matriz de features a propósito y eso es justo lo que la regla 5
prohíbe mezclar en una comparación de modelos.

Los scripts de Python van siempre con `uv run` (o con .venv activado): las
dependencias viven en .venv, no en el Python del sistema.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent

# El nombre con el que la fila entra a resultados.csv. Es "FLAML_arboles" y no
# "XGBoost" a propósito: con estimator_list de tres familias, el que gana puede
# no ser XGBoost, y la tabla de la tesina tiene que decir lo que realmente se
# ajustó. Cuál ganó queda escrito en las notas de cada corrida.
NOMBRE_MODELO = "FLAML_arboles"
NOMBRE_MODELO_REG = "FLAML_arboles_regresion"

# Familias que puede elegir FLAML. Agregar "extra_tree", "catboost" o
# "xgb_limitdepth" es cambiar esta lista y nada más, pero ojo: cuanto más ancho
# el espacio de búsqueda, más hay que justificar en la tesina que el modelo lo
# eligió un AutoML y no vos.
ESTIMADORES = ["xgboost", "lgbm", "rf"]

SEMILLA = 2026


# ---------------------------------------------------------------------------
# Convenio de carpetas: el mismo que dir_derivado()/dir_corrida() en R
# ---------------------------------------------------------------------------
def dir_derivado(corte: int, variante: str | None = None) -> Path:
    # Sin variante, la matriz completa (data/derived/2c/). Con variante, el
    # escalón de la ablación que exportó R/_exportar_ablacion.R
    # (data/derived/2c-e1_marcador/). Mismo convenio que dir_derivado() en
    # R/00_split.R.
    sufijo = f"-{variante}" if variante else ""
    return RAIZ / "data" / "derived" / f"{corte}c{sufijo}"


def dir_corrida(corte: int) -> Path:
    return RAIZ / "runs" / f"{corte}c"


def leer_split_txt(derived: Path) -> dict:
    meta = {}
    for linea in (derived / "SPLIT.txt").read_text(encoding="utf-8").splitlines():
        if ":" in linea:
            k, v = linea.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta


# ---------------------------------------------------------------------------
# Rolling origin, igual que rsample
# ---------------------------------------------------------------------------
class RollingOrigin:
    """Folds expandidos: cada corte entrena con todo el pasado y valida con el
    bloque siguiente, que es la situación de uso real.

    Replica rsample::rolling_origin(initial, assess, skip, cumulative = TRUE).

    OJO CON skip: en rsample el paso entre folds es `skip + 1`, no `skip`. Con
    initial=600, assess=100 y skip=100 los bloques de entrenamiento terminan en
    600, 701, 802, ... y no en 600, 700, 800. Está replicado tal cual para que
    los folds sean los mismos que venía usando el proyecto.

    Corta por POSICIÓN DE FILA, así que las filas tienen que venir ordenadas por
    fecha. Si no, un partido aplazado entra al bloque de entrenamiento de un
    fold que valida con partidos anteriores.

    DOS DETALLES DE FLAML, que no están documentados y se ven leyendo
    flaml/automl/task/generic_task.py (evaluate_model_CV):

    - Llama a get_n_splits() SIN argumentos y divide el presupuesto de tiempo
      por ese número. Si devuelve 0, revienta con ZeroDivisionError. Por eso la
      cantidad de filas se guarda en el constructor.
    - Si el splitter no expone un atributo `shuffle`, asume True y permuta las
      filas de entrenamiento de cada fold. No genera fuga (permuta el orden,
      no cambia qué filas entrenan), pero agrega azar innecesario a una
      validación que queremos exactamente reproducible. Con shuffle = False
      FLAML no toca nada.
    """

    def __init__(self, initial=600, assess=100, skip=100, fechas=None,
                 n_filas=None):
        self.initial = initial
        self.assess = assess
        self.paso = skip + 1
        self.fechas = fechas
        self.n_filas = n_filas if n_filas is not None else (
            len(fechas) if fechas is not None else 0)
        self.shuffle = False   # ver la nota de arriba: FLAML asume True

    def _cortes(self, n):
        fin = self.initial
        while fin + self.assess <= n:
            yield fin
            fin += self.paso

    def get_n_splits(self, X=None, y=None, groups=None):
        n = len(X) if X is not None else self.n_filas
        return sum(1 for _ in self._cortes(n))

    def split(self, X, y=None, groups=None):
        n = len(X)
        if n < self.initial + self.assess:
            raise ValueError(
                f"Train tiene {n} filas y no alcanza para initial={self.initial} "
                f"+ assess={self.assess}. Ajustá los parámetros del rolling origin."
            )
        for fin in self._cortes(n):
            idx_train = np.arange(0, fin)
            idx_val = np.arange(fin, fin + self.assess)
            # Misma comprobación explícita que hacía el script de R: ningún fold
            # puede entrenar con partidos posteriores al inicio de su bloque de
            # validación.
            if self.fechas is not None:
                if self.fechas[idx_train].max() > self.fechas[idx_val].min():
                    raise RuntimeError(
                        f"Fold que termina en la fila {fin}: hay partidos de "
                        "entrenamiento posteriores al inicio del bloque de "
                        "validación. ¿Las filas vienen ordenadas por fecha?"
                    )
            yield idx_train, idx_val


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
def cargar_split(derived: Path, corte: int, variante: str | None = None):
    if not derived.exists():
        que_correr = (f"Rscript R/_exportar_ablacion.R {corte}" if variante
                      else f"Rscript R/_exportar_split.R {corte}")
        sys.exit(
            f"No existe {derived}.\n"
            f"Corré primero:  {que_correr}"
        )

    X_train = pd.read_csv(derived / "X_train.csv")
    X_test = pd.read_csv(derived / "X_test.csv")
    y_train = pd.read_csv(derived / "y_train.csv")
    y_test = pd.read_csv(derived / "y_test.csv")

    if list(X_train.columns) != list(X_test.columns):
        sys.exit("X_train y X_test no tienen las mismas columnas. Reexportá el split.")

    meta = leer_split_txt(derived)
    corte_exportado = meta.get("corte_cuarto", "")
    if corte_exportado and corte_exportado != str(corte):
        sys.exit(
            f"El split exportado es de {corte_exportado}C pero pediste --corte "
            f"{corte}. Volvé a exportar con el mismo corte:  "
            f"Rscript R/_exportar_split.R {corte}"
        )

    print((derived / "SPLIT.txt").read_text(encoding="utf-8"))
    return X_train, X_test, y_train, y_test, meta


def preparar_matriz(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """One-hot para las columnas de texto y descarte de varianza cero.

    Es lo mismo que hacía la receta de R (step_dummy(one_hot = TRUE) + step_zv).
    Se codifica sobre la unión de train y test para que un club que sólo aparece
    en uno de los dos no genere columnas distintas en cada lado.

    TabPFN, en cambio, codifica los equipos como enteros: no es inconsistencia,
    es que TabPFN sólo acepta números y los trata como categóricos internamente.
    """
    from pandas.api.types import is_numeric_dtype

    cols_texto = [
        c for c in X_train.columns
        if not (is_numeric_dtype(X_train[c]) and is_numeric_dtype(X_test[c]))
    ]

    n_train = len(X_train)
    junto = pd.concat([X_train, X_test], axis=0, ignore_index=True)
    if cols_texto:
        junto = pd.get_dummies(junto, columns=cols_texto, dtype=float)

    tr = junto.iloc[:n_train].reset_index(drop=True)
    te = junto.iloc[n_train:].reset_index(drop=True)

    # step_zv: fuera las columnas que en TRAIN no varían. Se decide mirando sólo
    # train, nunca test.
    constantes = [c for c in tr.columns if tr[c].nunique(dropna=False) <= 1]
    if constantes:
        tr = tr.drop(columns=constantes)
        te = te.drop(columns=constantes)

    print(f"Features: {tr.shape[1]} columnas "
          f"({len(cols_texto)} categóricas expandidas, {len(constantes)} sin varianza descartadas)")
    return tr, te


def verificar_orden(y: pd.DataFrame, nombre: str) -> np.ndarray:
    """El rolling origin corta por posición, así que el orden por fecha es un
    supuesto del que depende la validez de la validación interna. Se verifica en
    vez de confiar."""
    fechas = pd.to_datetime(y["fecha"]).to_numpy()
    if not (fechas[:-1] <= fechas[1:]).all():
        sys.exit(
            f"{nombre} no viene ordenado por fecha. El split lo ordena en R; "
            "si esto falla, algo lo reordenó en el camino."
        )
    return fechas


# ---------------------------------------------------------------------------
# Guardado
# ---------------------------------------------------------------------------
def guardar_predicciones(runs: Path, modelo: str, y_test: pd.DataFrame,
                         pred_local: np.ndarray, prob_local=None, notas: str = ""):
    """Mismo formato que guardar_predicciones() de R, más un sidecar con la nota
    descriptiva. Las métricas las calcula R/_evaluar_predicciones.R."""
    runs.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "id_match": y_test["id_match"],
        "fecha": y_test["fecha"],
        "y_real": np.where(y_test["victoria_local"].to_numpy() == 1, "local", "visita"),
        "pred": np.where(pred_local, "local", "visita"),
        "prob_local": prob_local if prob_local is not None else np.nan,
    }).to_csv(runs / f"pred_{modelo}.csv", index=False, encoding="utf-8")

    (runs / f"pred_{modelo}.notas.txt").write_text(notas, encoding="utf-8")
    print(f"Predicciones en {runs / f'pred_{modelo}.csv'}")


def guardar_modelo(runs: Path, modelo: str, automl):
    """El modelo serializado y la configuración ganadora, por separado.

    La configuración va en texto plano a propósito: es lo que se cita en la
    tesina y tiene que poder leerse sin levantar Python (regla 9)."""
    config = {
        "estimador": automl.best_estimator,
        "config": automl.best_config,
        "log_loss_validacion": automl.best_loss,
        "segundos_hasta_el_mejor": automl.time_to_find_best_model,
        "estimadores_probados": ESTIMADORES,
        "corrida": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    (runs / f"config_{modelo}.json").write_text(
        json.dumps(config, indent=2, default=str), encoding="utf-8")

    try:
        import joblib
        joblib.dump(automl.model.estimator, runs / f"modelo_{modelo}.joblib")
    except Exception as e:  # noqa: BLE001
        print(f"AVISO: no pude serializar el modelo ({e}). La configuración "
              f"ganadora igual quedó en config_{modelo}.json.", file=sys.stderr)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corte", type=int, choices=(2, 3), default=2,
                    help="corte de cuarto: 2 (hasta el 2do, default) o 3. Tiene "
                         "que ser el mismo con el que corriste "
                         "Rscript R/_exportar_split.R")
    ap.add_argument("--variante", default=None,
                    help="escalón de la ablación (e1_marcador, e2_ritmo, ...). "
                         "Sin esto corre sobre la matriz completa. Lee de "
                         "data/derived/<corte>c-<variante>/ y guarda las "
                         "predicciones como FLAML_<variante>.")
    ap.add_argument("--presupuesto", type=int, default=300,
                    help="segundos de búsqueda de FLAML (default: 300)")
    ap.add_argument("--regresion", action="store_true",
                    help="además, ajustar sobre dif_puntos_final y umbralizar en 0")
    ap.add_argument("--initial", type=int, default=600,
                    help="filas del primer bloque de entrenamiento del rolling origin")
    ap.add_argument("--assess", type=int, default=100,
                    help="filas de cada bloque de validación")
    ap.add_argument("--skip", type=int, default=100,
                    help="salto entre folds (el paso real es skip + 1, como en rsample)")
    args = ap.parse_args()

    corte = args.corte
    variante = args.variante
    derived = dir_derivado(corte, variante)
    runs = dir_corrida(corte)

    # El nombre con el que se guardan predicciones, modelo e importancias. Cada
    # escalón de la ablación tiene el suyo para que no se pisen entre sí ni
    # pisen la corrida de la matriz completa.
    nombre = f"FLAML_{variante}" if variante else NOMBRE_MODELO
    nombre_reg = f"{nombre}_regresion" if variante else NOMBRE_MODELO_REG

    etiqueta = f" — escalón {variante}" if variante else ""
    print(f"\n=== FLAML ({'/'.join(ESTIMADORES)}) — corte {corte}C{etiqueta} ===")

    X_train, X_test, y_train, y_test, meta = cargar_split(derived, corte, variante)
    fechas_train = verificar_orden(y_train, "y_train")
    verificar_orden(y_test, "y_test")

    Xtr, Xte = preparar_matriz(X_train, X_test)
    ytr = y_train["victoria_local"].to_numpy()
    yte = y_test["victoria_local"].to_numpy()
    print(f"Train: {Xtr.shape}  Test: {Xte.shape}")

    from flaml import AutoML

    cv = RollingOrigin(initial=args.initial, assess=args.assess, skip=args.skip,
                       fechas=fechas_train)
    n_folds = cv.get_n_splits(Xtr)
    print(f"Cortes de rolling origin: {n_folds} "
          f"(initial={args.initial}, assess={args.assess}, paso={args.skip + 1})")
    if n_folds < 2:
        sys.exit("El rolling origin quedó con menos de 2 folds. Bajá --initial "
                 "o --assess.")

    # ---- clasificación ----
    automl = AutoML()
    automl.fit(
        X_train=Xtr, y_train=ytr,
        task="classification",
        metric="log_loss",          # calibración, no accuracy
        estimator_list=ESTIMADORES,
        eval_method="cv",
        split_type=cv,              # rolling origin, NO el CV aleatorio de FLAML
        time_budget=args.presupuesto,
        seed=SEMILLA,
        verbose=1,
    )

    print(f"\nGanó {automl.best_estimator} "
          f"(log loss de validación: {automl.best_loss:.4f})")
    print(f"Config: {automl.best_config}")

    idx_local = list(automl.classes_).index(1)
    prob = automl.predict_proba(Xte)[:, idx_local]
    pred_local = prob >= 0.5

    notas = (f"FLAML sobre {'/'.join(ESTIMADORES)}, tuneo por log_loss con "
             f"rolling origin expandido ({n_folds} folds, initial={args.initial}, "
             f"assess={args.assess}, paso={args.skip + 1}); presupuesto "
             f"{args.presupuesto}s; ganó {automl.best_estimator}; umbral 0.5")
    guardar_predicciones(runs, nombre, y_test, pred_local, prob, notas)
    guardar_modelo(runs, nombre, automl)

    # Orientativo nada más: la fila oficial la escribe R/_evaluar_predicciones.R.
    acc = float((pred_local == (yte == 1)).mean())
    print(f"\nAccuracy orientativa sobre los {len(yte)} partidos de test: {acc:.4f}")

    # Importancias, en CSV y no en gráfico: el mismo criterio que
    # importancia_TabPFN.csv. El gráfico se arma después, con los números.
    try:
        modelo_base = automl.model.estimator
        if hasattr(modelo_base, "feature_importances_"):
            imp = (pd.DataFrame({"variable": Xtr.columns,
                                 "importancia": modelo_base.feature_importances_})
                   .sort_values("importancia", ascending=False))
            imp.to_csv(runs / f"importancia_{nombre}.csv",
                       index=False, encoding="utf-8")
            print(f"\nTop 15 de importancia:\n{imp.head(15).to_string(index=False)}")
    except Exception as e:  # noqa: BLE001
        print(f"AVISO: no pude extraer importancias ({e}).", file=sys.stderr)

    # ---- regresión (opcional) ----
    # Mismo split y mismas features, prediciendo dif_puntos_final y umbralizando
    # en 0. Va como fila SEPARADA: es otro modelo, no otra forma de medir el
    # mismo.
    if args.regresion:
        print("\n=== Variante de regresión sobre dif_puntos_final ===")
        automl_r = AutoML()
        automl_r.fit(
            X_train=Xtr, y_train=y_train["dif_puntos_final"].to_numpy(),
            task="regression",
            metric="mae",
            estimator_list=ESTIMADORES,
            eval_method="cv",
            split_type=RollingOrigin(initial=args.initial, assess=args.assess,
                                     skip=args.skip, fechas=fechas_train),
            time_budget=args.presupuesto,
            seed=SEMILLA,
            verbose=1,
        )
        dif = automl_r.predict(Xte)
        mae = float(np.mean(np.abs(dif - y_test["dif_puntos_final"].to_numpy())))
        print(f"Ganó {automl_r.best_estimator} — MAE en test: {mae:.2f}")

        notas_r = (f"regresión sobre dif_puntos_final umbralizada en 0; FLAML "
                   f"sobre {'/'.join(ESTIMADORES)}, métrica mae, rolling origin "
                   f"expandido; ganó {automl_r.best_estimator}; MAE={mae:.2f}")
        # Sin prob_local: umbralizar una predicción de puntos no da una
        # probabilidad, y poner una inventada arruinaría el Brier de la tabla.
        guardar_predicciones(runs, nombre_reg, y_test, dif > 0,
                             None, notas_r)
        guardar_modelo(runs, nombre_reg, automl_r)

    print(f"\nListo. Ahora:  Rscript R/_evaluar_predicciones.R {corte}")


if __name__ == "__main__":
    main()
