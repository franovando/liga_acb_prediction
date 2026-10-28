"""
03_tabpfn.py — TabPFN sobre el MISMO split y las MISMAS features que los modelos de R.

Cambios respecto de script_tabpfn.py, y por qué:

1. NO arma features. Lee data/derived/X_train.csv y X_test.csv, que exporta R.
   Antes cada lenguaje armaba su propia matriz y las diferencias se colaban en
   la comparación (el script de R sacaba marcador_local_HT y marcador_visit_HT,
   el de Python los dejaba).
2. El token sale de la variable de entorno TABPFN_TOKEN. Antes estaba escrito
   en el código, dentro de un repo con origin público.
3. Guarda las predicciones en runs/<corte>c/pred_TabPFN.csv, con el mismo
   formato que guardar_predicciones() en R, y NO calcula métricas: las calcula
   R/_evaluar_predicciones.R llamando a evaluar(). Antes este script
   recalculaba accuracy, AUC y Brier por su cuenta "con el mismo formato que
   evaluar()", o sea dos implementaciones que nada obligaba a mantener
   iguales.
4. La importancia por permutación es opcional y NUNCA usa n_jobs=-1: TabPFN ya
   paraleliza internamente sobre PyTorch y forzar el paralelismo de sklearn
   encima produce un deadlock (13 minutos sin devolver nada, en la corrida vieja).

El corte (2do o 3er cuarto) se pasa como argumento, igual que en los scripts de
R, y tiene que ser el MISMO con el que exportaste el split. Si no coinciden,
este script corta con error en vez de mezclar los datos de un corte con los
resultados del otro.

Uso, desde la raíz del proyecto:

    Rscript R/_exportar_split.R 3        # genera data/derived/3c/
    export TABPFN_TOKEN=...              # en Windows: $env:TABPFN_TOKEN="..."
    python py/03_tabpfn.py --corte 3
    python py/03_tabpfn.py --corte 3 --importancia   # permutation importance (lento en CPU)
    Rscript R/_evaluar_predicciones.R 3  # de acá salen las métricas

Sin --corte corre el 2C, que es el default del proyecto.

En Colab con GPU esto tarda segundos; en CPU, minutos.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent

NOMBRE_MODELO = "TabPFN"


# Mismo convenio de carpetas por corte que dir_derivado() y dir_corrida() del
# lado de R. No hay variable de entorno ni constante global: el corte entra por
# --corte y se propaga como parámetro, así una corrida de 3C no puede terminar
# leyendo o escribiendo en la carpeta de 2C sin que se vea en el comando.
def dir_derivado(corte: int) -> Path:
    return RAIZ / "data" / "derived" / f"{corte}c"


def dir_corrida(corte: int) -> Path:
    return RAIZ / "runs" / f"{corte}c"


def leer_split_txt(derived: Path) -> dict:
    """Metadatos del corte, para que la fila de métricas quede autodescripta."""
    meta = {}
    for linea in (derived / "SPLIT.txt").read_text(encoding="utf-8").splitlines():
        if ":" in linea:
            k, v = linea.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta


def cargar_split(DERIVED: Path):
    if not DERIVED.exists():
        sys.exit(
            f"No existe {DERIVED}.\n"
            f"Corré primero:  Rscript R/_exportar_split.R {DERIVED.name[0]}"
        )

    X_train = pd.read_csv(DERIVED / "X_train.csv")
    X_test = pd.read_csv(DERIVED / "X_test.csv")
    y_train = pd.read_csv(DERIVED / "y_train.csv")
    y_test = pd.read_csv(DERIVED / "y_test.csv")

    if list(X_train.columns) != list(X_test.columns):
        sys.exit("X_train y X_test no tienen las mismas columnas. Reexportá el split.")

    print((DERIVED / "SPLIT.txt").read_text(encoding="utf-8"))
    # El corte lo decide R (R/00_split.R). Acá no se recorta ni se reordena nada.

    # TabPFN solo acepta números. Los equipos vienen como texto desde R; se
    # codifican con los niveles de la UNIÓN de train y test, para que el mismo
    # club tenga el mismo código en los dos conjuntos.
    #
    # Se detecta por "no es numérica" y no por dtype == object: en pandas 3 las
    # columnas de texto tienen dtype 'str' y la comparación contra object falla
    # en silencio, dejando los equipos como strings hasta que TabPFN revienta
    # con "could not convert string to float".
    from pandas.api.types import is_numeric_dtype
    for col in X_train.columns:
        if not (is_numeric_dtype(X_train[col]) and is_numeric_dtype(X_test[col])):
            niveles = sorted(set(X_train[col].astype(str)) | set(X_test[col].astype(str)))
            mapa = {v: i for i, v in enumerate(niveles)}
            X_train[col] = X_train[col].astype(str).map(mapa)
            X_test[col] = X_test[col].astype(str).map(mapa)

    assert X_train.notna().all().all(), "Hay NA en X_train"
    assert X_test.notna().all().all(), "Hay NA en X_test"

    return X_train, X_test, y_train, y_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corte", type=int, choices=(2, 3), default=2,
                    help="corte de cuarto: 2 (hasta el 2do, default) o 3 (hasta el 3ro). "
                         "Tiene que ser el mismo con el que corriste "
                         "Rscript R/_exportar_split.R")
    ap.add_argument("--importancia", action="store_true",
                    help="calcular permutation importance (lento en CPU)")
    ap.add_argument("--repeticiones", type=int, default=5)
    ap.add_argument("--modelo", default=os.getenv("TABPFN_MODEL", ""),
                    help="ruta o nombre del checkpoint de TabPFN")
    args = ap.parse_args()

    corte = args.corte
    DERIVED = dir_derivado(corte)
    RUNS = dir_corrida(corte)

    if not os.getenv("TABPFN_TOKEN"):
        print("AVISO: TABPFN_TOKEN no está seteada. Si el modelo es local, ignorá esto.",
              file=sys.stderr)

    print(f"\n=== TabPFN — corte {corte}C ===")

    X_train, X_test, y_train, y_test = cargar_split(DERIVED)

    # La guarda del corte va ANTES de entrenar: si el split exportado no es el
    # que pediste, no tiene sentido gastar los minutos de TabPFN para recién
    # después darse cuenta.
    meta = leer_split_txt(DERIVED)
    corte_exportado = meta.get("corte_cuarto", "")
    if corte_exportado and corte_exportado != str(corte):
        sys.exit(
            f"El split exportado es de {corte_exportado}C pero pediste --corte "
            f"{corte}. Volvé a exportar con el mismo corte antes de entrenar:  "
            f"Rscript R/_exportar_split.R {corte}"
        )

    yv_train = y_train["victoria_local"].to_numpy()
    yv_test = y_test["victoria_local"].to_numpy()

    print(f"Train: {X_train.shape}  Test: {X_test.shape}")

    from tabpfn import TabPFNClassifier

    clf = TabPFNClassifier(**({"model_path": args.modelo} if args.modelo else {}))
    clf.fit(X_train, yv_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    RUNS.mkdir(parents=True, exist_ok=True)

    # Predicciones, en el mismo formato que guardar_predicciones() de R, más un
    # sidecar con la nota descriptiva que después copia la fila de resultados.
    # Acá NO se calcula ninguna métrica: eso es trabajo de evaluar() en R.
    pd.DataFrame({
        "id_match": y_test["id_match"],
        "fecha": y_test["fecha"],
        "y_real": np.where(yv_test == 1, "local", "visita"),
        "pred": np.where(y_pred == 1, "local", "visita"),
        "prob_local": y_prob,
    }).to_csv(RUNS / f"pred_{NOMBRE_MODELO}.csv", index=False, encoding="utf-8")

    (RUNS / f"pred_{NOMBRE_MODELO}.notas.txt").write_text(
        f"clasificación, split y features exportados desde R"
        f"{'; checkpoint ' + args.modelo if args.modelo else ''}",
        encoding="utf-8")

    print(f"Predicciones en {RUNS / f'pred_{NOMBRE_MODELO}.csv'}")

    # Orientativo nada más, para no quedarse sin señal mientras corre: la fila
    # con accuracy, AUC y Brier la escribe R/_evaluar_predicciones.R.
    acc = float((y_pred == yv_test).mean())
    print(f"Accuracy orientativa sobre los {len(yv_test)} partidos de test: {acc:.4f}")

    if args.importancia:
        from sklearn.inspection import permutation_importance
        print(f"\nPermutation importance ({X_test.shape[1]} columnas x "
              f"{args.repeticiones} repeticiones). Sin n_jobs: TabPFN ya "
              "paraleliza y forzarlo cuelga el proceso.")
        r = permutation_importance(clf, X_test, yv_test,
                                   n_repeats=args.repeticiones, random_state=42)
        imp = (pd.DataFrame({"variable": X_test.columns,
                             "importancia": r.importances_mean,
                             "sd": r.importances_std})
               .sort_values("importancia", ascending=False))
        imp.to_csv(RUNS / "importancia_TabPFN.csv", index=False, encoding="utf-8")
        print(imp.head(15).to_string(index=False))

    print(f"\nListo. Ahora:  Rscript R/_evaluar_predicciones.R {corte}")


if __name__ == "__main__":
    main()
