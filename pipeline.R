# =============================================================================
# pipeline.R — Todos los datos de la tesina, de la API de la ACB a Python
# =============================================================================
# Desde la raíz del repo:
#
#   Rscript pipeline.R                 # todo, desde cero: baja la API de la ACB
#                                      # (necesita ACB_TOKEN, ver .Renviron.example)
#   Rscript pipeline.R --no-scraping   # usa los PBP_Data que ya están en
#                                      # datos/pbp/; tarda segundos
#
# --no-scraping es lo que se corre cuando cambia etl/04_box_score.R,
# etl/05_datos.R o etl/06_split.R. Las temporadas están en TEMPORADAS
# (etl/05_datos.R).
# =============================================================================

source("etl/06_split.R")   # carga también etl/05_datos.R
source("etl/04_box_score.R")

NO_SCRAPING <- "--no-scraping" %in% commandArgs(trailingOnly = TRUE)

DIR_PBP        <- file.path("datos", "pbp")
DIR_BOX_SCORES <- file.path("datos", "box_scores")
RUTA_RDS       <- file.path(DIR_PBP, sprintf("PBP_Data_%s.rds", ETIQUETA_TEMPORADAS))

if (!NO_SCRAPING && !nzchar(Sys.getenv("ACB_TOKEN"))) {
  stop("Falta ACB_TOKEN. Copiá .Renviron.example a .Renviron en la raíz del ",
       "repo y completalo, o corré con --no-scraping si ya tenés los ",
       "PBP_Data en datos/pbp/.")
}

cat("\nTemporadas ", min(TEMPORADAS), "-", max(TEMPORADAS), ", ",
    if (NO_SCRAPING) "sin scraping" else "desde la API", "\n", sep = "")

# --- scraping y limpieza (01, 03) ---------------------------------------------
if (!NO_SCRAPING) {
  source("etl/config/seasons.R")
  source("etl/01_scrape.R")
  source("etl/03_clean.R")

  dir.create(DIR_PBP, showWarnings = FALSE, recursive = TRUE)

  for (temporada in TEMPORADAS) {
    # Los CSV partido a partido van a una carpeta temporal, se juntan en el
    # PBP_Data de la temporada y se borran.
    carpeta <- file.path(tempdir(), temporada)
    scrape_season(temporada, carpeta)
    clean_pbp(temporada, carpeta,
              file.path(DIR_PBP, sprintf("PBP_Data_%d.csv", temporada)))
    unlink(carpeta, recursive = TRUE)
  }

  # Con datos nuevos, la caché vieja deja de valer.
  if (file.exists(RUTA_RDS)) unlink(RUTA_RDS)
}

# --- play-by-play de todas las temporadas (04) --------------------------------
# El .rds es sólo una caché: si no está, se rearma desde los PBP_Data.
if (file.exists(RUTA_RDS)) {
  cat("-> Leyendo", RUTA_RDS, "\n")
  pbp <- readRDS(RUTA_RDS)
} else {
  pbp <- consolidar_pbp(TEMPORADAS, dir_pbp = DIR_PBP, guardar = RUTA_RDS)
}

# --- box score al 2C (04) -----------------------------------------------------
escribir_box_scores(pbp, cortes = 2L, dir_salida = DIR_BOX_SCORES,
                    etiqueta = ETIQUETA_TEMPORADAS)
rm(pbp)

# --- matriz de features y split para Python (05, 06) --------------------------
sp <- describir_split(split_por_temporada(cargar_box_score(2L),
                                          TEMPORADA_VALIDACION,
                                          TEMPORADA_TEST))
exportar_split(sp)

cat("\nListo. Los datos para los notebooks están en datos/matrices/.\n")
