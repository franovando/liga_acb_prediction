# =============================================================================
# config/seasons.R — Temporadas de la ACB y acceso a la API
# =============================================================================
# pipeline.R le hace source() antes de bajar la API.
# =============================================================================

library(tibble)

# season_id: año en que termina la temporada (2022 = 2021-22).
# edition_id: id de la temporada en la API.
SEASONS <- tibble::tribble(
  ~season_id, ~season_name,   ~edition_id,
  2017,       "2016-2017",    81,
  2018,       "2017-2018",    82,
  2019,       "2018-2019",    83,
  2020,       "2019-2020",    84,
  2021,       "2020-2021",    85,
  2022,       "2021-2022",    86,
  2023,       "2022-2023",    87,
  2024,       "2023-2024",    88,
  2025,       "2024-2025",    89,
  2026,       "2025-2026",    90
)

API_CONFIG <- list(
  pbp_url = "https://api2.acb.com/api/v1/openapilive/PlayByPlay/matchevents",
  weeks_url_template = "https://api2.acb.com/api/v1/openapilive/Matchweeks/lite?idCompetition=1&idEdition={edition_id}",
  matches_url_template = "https://api2.acb.com/api/v1/openapilive/Matches/matchesbymatchweeklite?idCompetition=1&idEdition={edition_id}&idMatchweek=",
  # El token sale de ACB_TOKEN (ver .Renviron.example).
  bearer_token = paste("Bearer", Sys.getenv("ACB_TOKEN"))
)

get_season_config <- function(season_id) {
  as.list(SEASONS[SEASONS$season_id == season_id, ])
}
