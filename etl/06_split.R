# ==============================================================================
# 06_split.R — Definición ÚNICA de los conjuntos de train, validación y test
# ==============================================================================
# Ningún otro script define un split. Son temporadas enteras, en orden:
#   train      : todas las temporadas anteriores a TEMPORADA_VALIDACION
#   validacion : TEMPORADA_VALIDACION entera; ahí se comparan los modelos
#   test       : TEMPORADA_TEST entera; una sola vez, para el modelo elegido
# ==============================================================================

library(dplyr)

source("etl/05_datos.R")

# Año de inicio de la temporada: 2024 = temporada 2024-25.
TEMPORADA_VALIDACION <- 2024L
TEMPORADA_TEST       <- 2025L

split_por_temporada <- function(df, temporada_validacion, temporada_test) {
  sp <- list(
    train                = df %>% filter(temporada <  temporada_validacion),
    validacion           = df %>% filter(temporada == temporada_validacion),
    test                 = df %>% filter(temporada == temporada_test),
    temporada_validacion = temporada_validacion,
    temporada_test       = temporada_test,
    corte_cuarto         = df$corte_cuarto[1]
  )
  if (max(sp$train$fecha) >= min(sp$validacion$fecha) ||
      max(sp$validacion$fecha) >= min(sp$test$fecha)) {
    stop("El split no es cronológico: train, validación y test tienen que ",
         "ir en ese orden, sin superponerse.")
  }
  sp
}

describir_split <- function(sp) {
  cat("\n--- Split al ", sp$corte_cuarto, "C: validación ", sp$temporada_validacion,
      ", test ", sp$temporada_test, " ---\n", sep = "")
  for (conj in c("train", "validacion", "test")) {
    d <- sp[[conj]]
    cat(sprintf("%-16s: %d partidos ( %s a %s ) — %d local / %d visita (tasa base local %.4f)\n",
                conj, nrow(d), format(min(d$fecha)), format(max(d$fecha)),
                sum(d$victoria_local), sum(1 - d$victoria_local),
                mean(d$victoria_local)))
  }
  invisible(sp)
}

# Lo que lee Python: datos/matrices/matriz_<corte>c.csv, una fila por partido,
# ordenadas por fecha. Primero id_match, fecha, temporada, jornada, conjunto
# ("train", "validacion" o "test"), victoria_local (1 = ganó el local) y
# dif_puntos_final; después todas las features de construir_features().
exportar_split <- function(sp) {
  armar <- function(d, conjunto) {
    bind_cols(
      d %>% transmute(id_match, fecha = format(fecha), temporada, jornada,
                      conjunto = conjunto, victoria_local, dif_puntos_final),
      construir_features(d)
    )
  }
  m <- bind_rows(armar(sp$train, "train"), armar(sp$validacion, "validacion"),
                 armar(sp$test, "test"))

  archivo <- file.path("datos", "matrices", sprintf("matriz_%dc.csv", sp$corte_cuarto))
  dir.create(dirname(archivo), showWarnings = FALSE, recursive = TRUE)
  utils::write.csv(m, archivo, row.names = FALSE, fileEncoding = "UTF-8")

  message("Exportado ", archivo, ": ", nrow(sp$train), " train + ",
          nrow(sp$validacion), " validación + ", nrow(sp$test), " test, ",
          ncol(m) - 7L, " features")
  invisible(archivo)
}
