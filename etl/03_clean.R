# =============================================================================
# 03_clean.R — Limpieza del play-by-play de una temporada
# =============================================================================
# clean_pbp(temporada, carpeta, archivo) junta los CSV partido a partido que
# dejó scrape_season() en `carpeta`, agrega las columnas de eventos que usa
# 04_box_score.R y escribe `archivo` (datos/pbp/PBP_Data_<temporada>.csv).
# =============================================================================

library(dplyr)

# Una columna 0/1 (o 0/1/2) por tipo de evento, a partir de type.description.
add_boxscore_columns <- function(df) {

  # En la 2021-22 la ACB cargaba las faltas en bonus como "Personal no TL"
  # aunque dieran dos libres; desde la 2022-23 son "Personal 2TL". Se
  # reetiquetan las "no TL" cuyo siguiente evento de juego es un tiro libre.
  NEUTROS <- c("Falta recibida", "Entra Pista", "Sale Pista", "Tiempo Muerto",
               "Tiempo de Televisión", "Asistencia Falta Recibida",
               "Tiempo Instant Replay")
  df <- df %>%
    group_by(id_match) %>%
    arrange(order, .by_group = TRUE) %>%
    mutate(siguiente = lead(if_else(
      type.description %in% NEUTROS | startsWith(type.description, "IR - "),
      NA_character_, type.description))) %>%
    tidyr::fill(siguiente, .direction = "up") %>%
    mutate(type.description = if_else(
      type.description == "Personal no TL" &
        siguiente %in% c("Canasta de 1", "Intento fallado de 1"),
      "Personal 2TL", type.description)) %>%
    ungroup() %>%
    select(-siguiente)

  df <- df %>%
    mutate(
      reb_def = as.integer(type.description == "Rebote Defensivo"),
      reb_of = as.integer(type.description == "Rebote Ofensivo"),
      asis_t2 = as.integer(type.description == "Asistencia tiro de 2"),
      asis_t3 = as.integer(type.description == "Asistencia tiro de 3"),
      recuperacion = as.integer(type.description == "Recuperación"),
      tapon = as.integer(type.description == "Tapón"),
      perdida = as.integer(type.description == "Pérdida"),
      falta = as.integer(type.description %in% c(
        "Personal 2TL", "Personal 1TL", "Falta en Ataque",
        "Personal no TL", "Personal TL Comp", "Personal 3TL",
        "Antidep 1TL", "Antidep 2TL", "Técnica 1TL", "Técnica 2TL"
      )),
      T1A = as.integer(type.description == "Canasta de 1"),
      T1F = as.integer(type.description == "Intento fallado de 1"),
      T2A = as.integer(type.description == "Canasta de 2"),
      T2F = as.integer(type.description == "Intento fallado de 2"),
      T3A = as.integer(type.description == "Canasta de 3"),
      T3F = as.integer(type.description == "Intento fallado de 3"),
      mateA = as.integer(type.description == "Mate"),
      mateF = as.integer(type.description == "Mate fuera")
    )

  # Los mates cuentan como tiros de 2
  df <- df %>%
    mutate(
      T2A = T2A + mateA,
      T2F = T2F + mateF,
      T1I = T1A + T1F,
      T2I = T2A + T2F,
      T3I = T3A + T3F,
      asistencias = asis_t2 + asis_t3
    )

  # Canastas de transición (después de una recuperación) y de segunda
  # oportunidad (después de un rebote ofensivo del mismo equipo)
  df <- df %>%
    group_by(id_match) %>%
    arrange(order, .by_group = TRUE) %>%
    mutate(
      prev_team = lag(team.team_actual_name),
      prev_was_steal = lag(recuperacion, default = 0) > 0,
      prev_was_oreb = lag(reb_of, default = 0) > 0 & prev_team == team.team_actual_name,
      transition_fgm2 = as.integer(T2A > 0 & prev_was_steal),
      transition_fgm3 = as.integer(T3A > 0 & prev_was_steal),
      second_chance_fgm2 = as.integer(T2A > 0 & prev_was_oreb),
      second_chance_fgm3 = as.integer(T3A > 0 & prev_was_oreb)
    ) %>%
    ungroup() %>%
    select(-prev_team, -prev_was_steal, -prev_was_oreb)

  df %>% arrange(order, id_match)
}

clean_pbp <- function(season_id, carpeta, archivo) {
  archivos <- list.files(carpeta, pattern = "*.csv", full.names = TRUE)

  # Todo como texto (colClasses = "character"), para que bind_rows no falle
  # por tipos distintos entre partidos; después type.convert devuelve los
  # números. Un archivo ilegible corta acá.
  df <- lapply(archivos, read.csv, encoding = "UTF-8",
               stringsAsFactors = FALSE, colClasses = "character") %>%
    bind_rows() %>%
    type.convert(as.is = TRUE)

  df <- add_boxscore_columns(df) %>% mutate(season_id = .env$season_id)
  write.csv(df, archivo, row.names = FALSE, fileEncoding = "UTF-8")

  cat("Limpieza ", season_id, ": ", length(archivos), " partidos, ",
      format(nrow(df), big.mark = ".", decimal.mark = ","), " eventos -> ", archivo, "\n", sep = "")
}
