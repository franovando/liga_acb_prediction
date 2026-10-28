# =============================================================================
# 01_scrape.R — Descarga del play-by-play de una temporada
# =============================================================================
# scrape_season(temporada, carpeta) baja de la API de la ACB un CSV por partido
# terminado. Usa API_CONFIG y get_season_config() de config/seasons.R.
# =============================================================================

library(jsonlite)
library(httr)
library(dplyr)
library(foreach)
library(doParallel)

scrape_season <- function(season_id, carpeta) {
  season <- get_season_config(season_id)
  cat("\n=== Scraping", season$season_name, "===\n")
  dir.create(carpeta, recursive = TRUE, showWarnings = FALSE)

  # Jornadas y partidos de la temporada
  weeks_url <- gsub("\\{edition_id\\}", season$edition_id, API_CONFIG$weeks_url_template)
  matches_url <- gsub("\\{edition_id\\}", season$edition_id, API_CONFIG$matches_url_template)

  week_response <- GET(weeks_url, add_headers(Authorization = API_CONFIG$bearer_token))
  stop_for_status(week_response)
  weeks_json <- fromJSON(content(week_response, type = "text", encoding = "UTF-8"), flatten = TRUE)
  weeks <- tibble(id = weeks_json$id, jornada = weeks_json$descriptor)

  match_response <- GET(matches_url, add_headers(Authorization = API_CONFIG$bearer_token))
  stop_for_status(match_response)
  match_json <- fromJSON(content(match_response, type = "text", encoding = "UTF-8"), flatten = TRUE)

  all_matches <- tibble(
    match_id = match_json$id,
    team_local = match_json$local_team.team_actual_name,
    fecha = as.Date(as.POSIXct(match_json$date, origin="1970-01-01", tz="Europe/Madrid")),
    hora = format(as.POSIXct(match_json$time, origin="1970-01-01", tz="Europe/Madrid"), "%H:%M"),
    team_visitor = match_json$visitor_team.team_actual_name,
    score_local = match_json$score_local,
    score_visitor = match_json$score_visitor,
    finalized = match_json$finalized,
    weekid = match_json$id_matchweek
  ) %>%
    inner_join(weeks, by = c("weekid" = "id")) %>%
    filter(finalized == TRUE) %>%
    mutate(jornada = trimws(gsub("Jornada ", "", jornada)),
           # El "/" de "1/4 de final" rompía el nombre del archivo
           jornada = gsub("1/4 de final", "Cuartos", jornada))

  match_ids <- unique(all_matches$match_id)
  n_cores <- parallel::detectCores() - 1
  cat("Bajando", length(match_ids), "partidos terminados con", n_cores, "núcleos...\n")

  cl <- parallel::makeCluster(n_cores)
  on.exit(parallel::stopCluster(cl))
  doParallel::registerDoParallel(cl)

  # Un partido por tarea. Si uno falla después de los reintentos, foreach corta.
  foreach(
    match_id = match_ids,
    .packages = c("httr", "jsonlite", "dplyr"),
    .export = "API_CONFIG"
  ) %dopar% {
    pbp_url <- paste0(API_CONFIG$pbp_url, "?idMatch=", match_id, "&jvFilter=true")
    # Hasta 4 intentos, con esperas de entre 1 y 10 segundos
    pbp_response <- httr::RETRY("GET", pbp_url,
                                add_headers(Authorization = API_CONFIG$bearer_token),
                                times = 4, pause_base = 1, pause_cap = 10)
    stop_for_status(pbp_response)

    pbp_data <- fromJSON(content(pbp_response, type = "text", encoding = "UTF-8"),
                         flatten = TRUE)
    if (!is.data.frame(pbp_data)) pbp_data <- Filter(is.data.frame, pbp_data)[[1]]
    pbp_data <- pbp_data %>% select(-any_of(c("license.media", "team.media")))

    match_info <- all_matches %>% filter(match_id == .env$match_id)

    # Local y visitante: el nombre (y la abreviatura) más frecuente de cada lado
    team_summary <- pbp_data %>%
      filter(!is.na(team.team_actual_name), !is.na(local)) %>%
      group_by(team.team_actual_name, local) %>%
      summarise(n = n(), .groups = "drop") %>%
      slice_max(n, n = 1, by = local, with_ties = FALSE)

    team_local <- as.character(team_summary$team.team_actual_name[team_summary$local == TRUE][1])
    team_visitor <- as.character(team_summary$team.team_actual_name[team_summary$local == FALSE][1])

    team_abbrev <- pbp_data %>%
      filter(!is.na(team.team_abbrev_name), !is.na(local)) %>%
      count(team.team_abbrev_name, local) %>%
      slice_max(n, n = 1, by = local, with_ties = FALSE)

    team_local_abb <- as.character(team_abbrev$team.team_abbrev_name[team_abbrev$local == TRUE][1])
    team_visitor_abb <- as.character(team_abbrev$team.team_abbrev_name[team_abbrev$local == FALSE][1])

    pbp_data <- pbp_data %>%
      mutate(
        jornada  = match_info$jornada[1],
        team     = if_else(local, team_local, team_visitor),
        date     = match_info$fecha[1],
        time     = match_info$hora[1],
        opponent = if_else(local, team_visitor, team_local),
        score_local_final   = match_info$score_local[1],
        score_visitor_final = match_info$score_visitor[1]
      )

    # El match_id en el nombre evita que se pisen dos partidos de la misma jornada
    filename <- paste0("J", match_info$jornada[1], "_", match_id, "_", team_local_abb, "_", team_visitor_abb, "_PBP.csv")
    write.csv(pbp_data, file.path(carpeta, filename), row.names = FALSE, fileEncoding = "UTF-8")

    Sys.sleep(0.25)  # pausa por partido, para no saturar la API
    NULL
  }

  cat("Listo:", length(list.files(carpeta)), "archivos en", carpeta, "\n")
}
