# =============================================================================
# 04_box_score.R — Del play-by-play limpio a una fila por partido
# =============================================================================
# consolidar_pbp() junta los PBP_Data_<temporada>.csv de datos/pbp/.
# escribir_box_scores() los agrega a una fila por partido, con todo acumulado
# hasta el final del cuarto de corte, y deja datos/box_scores/.
# =============================================================================

library(dplyr)
library(tidyr)

# Estadísticas que se suman por equipo: nombre en el box score = columna del
# play-by-play. Se declaran una sola vez para los dos equipos (local_, visita_).
ESTADISTICAS <- c(
  T2A            = "T2A",
  T2I            = "T2I",
  T3A            = "T3A",
  T3I            = "T3I",
  T1A            = "T1A",
  T1I            = "T1I",
  reb_of         = "reb_of",
  reb_def        = "reb_def",
  perdida        = "perdida",
  falta          = "falta",
  asistencias    = "asistencias",
  tapon          = "tapon",
  recuperacion   = "recuperacion",
  `2nd_chance2`  = "second_chance_fgm2",
  `2nd_chance3`  = "second_chance_fgm3",
  transicion2    = "transition_fgm2",
  transicion3    = "transition_fgm3"
)

# Junta los PBP_Data de las temporadas y los guarda en `guardar` (.rds, caché).
consolidar_pbp <- function(temporadas, dir_pbp, guardar) {
  archivos <- file.path(dir_pbp, sprintf("PBP_Data_%d.csv", temporadas))
  pbp <- bind_rows(lapply(archivos, utils::read.csv, encoding = "UTF-8",
                          stringsAsFactors = FALSE))
  saveRDS(pbp, guardar)
  cat("-> ", length(archivos), " temporadas, ",
      format(nrow(pbp), big.mark = ".", decimal.mark = ","), " eventos, ",
      length(unique(pbp$id_match)), " partidos -> ", guardar, "\n", sep = "")
  pbp
}

# Marcador al cierre de cada cuarto: el del ÚLTIMO evento del cuarto según
# `order`. No sirve max(): la ACB carga eventos con marcadores erróneos (a veces
# el resultado final en pleno 1er cuarto).
cierres_de_cuarto <- function(datos) {
  datos %>%
    group_by(id_match, period) %>%
    slice_max(order, n = 1, with_ties = FALSE) %>%
    ungroup() %>%
    transmute(id_match, period, score_local, score_visitor,
              dif = score_local - score_visitor)
}

# Una fila por partido, con todo acumulado hasta el final del cuarto `corte`.
armar_box_score <- function(pbp, corte) {

  # Todo sale de los eventos hasta el corte, así que no hay fuga del resto del
  # partido. Las únicas columnas que miran el final son los dos targets.
  datos <- pbp %>% filter(period <= corte)

  # which(local) y no local == TRUE: hay eventos sin equipo, con local = NA.
  cabecera <- datos %>%
    group_by(id_match) %>%
    summarise(
      equipo_local     = first(team[which(local)]),
      equipo_visitante = first(team[which(!local)]),
      jornada = first(jornada),
      fecha   = first(date),
      hora    = first(time),
      victoria_local   = as.integer(first(score_local_final) >
                                    first(score_visitor_final)),
      dif_puntos_final = first(score_local_final) - first(score_visitor_final),
      .groups = "drop"
    )

  cierres <- cierres_de_cuarto(datos)

  # Marcador al corte: marcador_local_Q<corte> y marcador_visit_Q<corte>
  col_local <- paste0("marcador_local_Q", corte)
  col_visit <- paste0("marcador_visit_Q", corte)
  marcador <- cierres %>%
    filter(period == corte) %>%
    transmute(id_match, !!col_local := score_local, !!col_visit := score_visitor)

  # Cronología: la diferencia acumulada al cierre de cada cuarto,
  # dif_puntos_Q1 ... dif_puntos_Q<corte>. La última es la diferencia al corte.
  cols_cronologia <- paste0("dif_puntos_Q", seq_len(corte))
  cronologia <- cierres %>%
    transmute(id_match, cuarto = paste0("dif_puntos_Q", period), dif) %>%
    pivot_wider(names_from = cuarto, values_from = dif) %>%
    select(id_match, all_of(cols_cronologia))

  # Estadísticas por equipo: se suman en formato largo (una fila por partido y
  # rol) y después se ensanchan a local_* / visita_*.
  estadisticas <- datos %>%
    filter(!is.na(local)) %>%
    mutate(rol = if_else(local, "local", "visita")) %>%
    group_by(id_match, rol) %>%
    summarise(across(all_of(unname(ESTADISTICAS)),
                     ~ sum(.x, na.rm = TRUE)), .groups = "drop") %>%
    rename(!!!stats::setNames(unname(ESTADISTICAS), names(ESTADISTICAS))) %>%
    pivot_wider(names_from  = rol,
                values_from = all_of(names(ESTADISTICAS)),
                names_glue  = "{rol}_{.value}",
                values_fill = 0)

  box <- cabecera %>%
    inner_join(marcador,     by = "id_match") %>%
    inner_join(cronologia,   by = "id_match") %>%
    inner_join(estadisticas, by = "id_match")

  # Primero todas las estadísticas del local y después todas las de la visita
  box %>%
    select(
      id_match, equipo_local, equipo_visitante, jornada, fecha, hora,
      victoria_local, dif_puntos_final,
      all_of(c(col_local, col_visit, cols_cronologia)),
      all_of(paste0("local_",  names(ESTADISTICAS))),
      all_of(paste0("visita_", names(ESTADISTICAS)))
    ) %>%
    arrange(fecha, id_match)
}

# Escribe datos/box_scores/box_score_<corte>c_<etiqueta>.csv para cada corte.
escribir_box_scores <- function(pbp, cortes, dir_salida, etiqueta) {
  dir.create(dir_salida, showWarnings = FALSE, recursive = TRUE)
  for (corte in cortes) {
    box  <- armar_box_score(pbp, corte)
    ruta <- file.path(dir_salida, sprintf("box_score_%dc_%s.csv", corte, etiqueta))
    utils::write.csv(box, ruta, row.names = FALSE, fileEncoding = "UTF-8")
    cat("-> Box score al ", corte, "C: ", nrow(box), " partidos x ", ncol(box),
        " columnas -> ", ruta, "\n", sep = "")
  }
}
