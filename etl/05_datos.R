# ==============================================================================
# 05_datos.R — Carga del box score y definición ÚNICA de la matriz de features
# ==============================================================================
# El único lugar del proyecto donde se decide qué columnas pueden usar los
# modelos.
# ==============================================================================

library(dplyr)
library(tidyr)

# ------------------------------------------------------------------------------
# Nombres de equipos
# ------------------------------------------------------------------------------
# La API cambia el nombre de los clubes con cada sponsor, y a veces usa
# abreviaturas. Este es el único diccionario de nombres: todo nombre que venga
# de la API tiene que estar acá, y canonizar_equipo() corta con error ante uno
# nuevo.
EQUIPOS_CANONICOS <- c(
  "Bitci Baskonia"          = "Baskonia",
  "Cazoo Baskonia"          = "Baskonia",
  "Kosner Baskonia"         = "Baskonia",
  "Baskonia"                = "Baskonia",
  "Gran Canaria"            = "Gran Canaria",
  "Dreamland Gran Canaria"  = "Gran Canaria",
  "Lenovo Tenerife"         = "Tenerife",
  "La Laguna Tenerife"      = "Tenerife",
  "Coosur Real Betis"       = "Real Betis",
  "Real Betis Baloncesto"   = "Real Betis",
  "Urbas Fuenlabrada"       = "Fuenlabrada",
  "Carplus Fuenlabrada"     = "Fuenlabrada",
  "Carplus Fuenla"          = "Fuenlabrada",
  "BAXI Manresa"            = "Manresa",
  "Barça"                   = "Barcelona",
  "Bàsquet Girona"          = "Girona",
  "Casademont Zaragoza"     = "Zaragoza",
  "Casademont Zgz"          = "Zaragoza",
  "Coviran Granada"         = "Granada",
  "Hereda San Pablo Burgos" = "Burgos",
  "Recoletas Salud San Pablo Burgos" = "Burgos",
  "Hiopos Lleida"           = "Lleida",
  "Joventut Badalona"       = "Joventut",
  "Joventut"                = "Joventut",
  "Club Joventut Badalona"  = "Joventut",
  "Asisa Joventut"          = "Joventut",
  "Leyma Coruña"            = "Coruña",
  "Monbus Obradoiro"        = "Obradoiro",
  "Monbus Obra"             = "Obradoiro",
  "MoraBanc Andorra"        = "Andorra",
  "Real Madrid"             = "Real Madrid",
  "Río Breogán"             = "Breogán",
  "Surne Bilbao Basket"     = "Bilbao",
  "Surne Bilbao"            = "Bilbao",
  "UCAM Murcia"             = "Murcia",
  "UCAM Murcia CB"          = "Murcia",
  "Unicaja"                 = "Unicaja",
  "Valencia Basket"         = "Valencia",
  "Valencia Basket Club"    = "Valencia",
  "Zunder Palencia"         = "Palencia"
)

canonizar_equipo <- function(x) {
  x   <- trimws(as.character(x))
  out <- unname(EQUIPOS_CANONICOS[x])
  if (any(is.na(out))) {
    stop("Equipos sin entrada en EQUIPOS_CANONICOS: ",
         paste(sort(unique(x[is.na(out)])), collapse = ", "),
         "\nAgregalos al diccionario en etl/05_datos.R antes de seguir.")
  }
  out
}

# Las temporadas del proyecto (season_id: 2022 = temporada 2021-22) y su tramo,
# "21-26", que va en el nombre del rds y de los box scores.
TEMPORADAS <- 2022:2026
ETIQUETA_TEMPORADAS <- sprintf("%02d-%02d", (min(TEMPORADAS) - 1) %% 100,
                               max(TEMPORADAS) %% 100)

# Columnas que nunca entran en X, además de las del marcador que dependen del
# corte (ver construir_features()). La fecha no va como predictor: con un split
# cronológico, las fechas de validación y test caen fuera del rango del train.
COLS_EXCLUIDAS <- c(
  "id_match", "fecha", "hora", "jornada", "temporada",
  "victoria_local", "dif_puntos_final", "corte_cuarto"
)

# partidos_jugados_local y _visita: partidos que el equipo ya jugó en la
# temporada, sin contar este (el primero vale 0). Mira sólo hacia atrás, así
# que se calcula antes de partir los datos sin que haya fuga.
agregar_partidos_jugados <- function(df) {
  jugados <- bind_rows(
    df %>% transmute(id_match, fecha, temporada, equipo = equipo_local,     rol = "local"),
    df %>% transmute(id_match, fecha, temporada, equipo = equipo_visitante, rol = "visita")
  ) %>%
    arrange(equipo, fecha, id_match) %>%
    group_by(equipo, temporada) %>%
    mutate(partidos_jugados = row_number() - 1L) %>%
    ungroup() %>%
    select(id_match, rol, partidos_jugados) %>%
    pivot_wider(names_from = rol, values_from = partidos_jugados,
                names_prefix = "partidos_jugados_", names_sort = TRUE)

  df %>%
    left_join(jugados, by = "id_match") %>%
    arrange(fecha, id_match)
}

# Récord y puntos a favor y en contra de la fase regular 2020-21 (tabla final
# de la ACB), la temporada anterior a la primera del proyecto, que no se
# scrapea. Los de las temporadas siguientes salen de los datos.
RECORD_2020 <- tribble(
  ~equipo,         ~victorias, ~derrotas, ~puntos_favor, ~puntos_contra,
  "Real Madrid",   34,  2, 3132, 2741,
  "Barcelona",     32,  4, 3162, 2621,
  "Tenerife",      27,  9, 3147, 2861,
  "Valencia",      24, 12, 3107, 2917,
  "Baskonia",      23, 13, 2952, 2814,
  "Burgos",        22, 14, 3130, 2995,
  "Joventut",      20, 16, 3089, 3070,
  "Gran Canaria",  18, 18, 2931, 3004,
  "Andorra",       17, 19, 2850, 2866,
  "Manresa",       17, 19, 3017, 3134,
  "Unicaja",       17, 19, 3014, 2997,
  "Murcia",        16, 20, 2924, 2951,
  "Zaragoza",      14, 22, 3127, 3189,
  "Obradoiro",     12, 24, 2888, 3015,
  "Fuenlabrada",   12, 24, 2920, 3019,
  "Real Betis",    11, 25, 2786, 3032,
  "Bilbao",        10, 26, 2880, 3118,
  "Estudiantes",    9, 27, 2969, 3185,
  "San Sebastián",  7, 29, 2650, 3146
)

# pct_victorias_anterior_local y _visita: porcentaje de victorias del equipo en
# la fase regular de la temporada anterior. NA si no la jugó (recién ascendido).
agregar_temporada_anterior <- function(df) {
  regular <- df %>% filter(es_playoff == 0)
  anterior <- bind_rows(
    regular %>% transmute(temporada, equipo = equipo_local,     gano = victoria_local == 1),
    regular %>% transmute(temporada, equipo = equipo_visitante, gano = victoria_local == 0)
  ) %>%
    group_by(temporada, equipo) %>%
    summarise(pct = mean(gano), .groups = "drop") %>%
    bind_rows(RECORD_2020 %>% transmute(temporada = 2020L, equipo,
                                        pct = victorias / (victorias + derrotas))) %>%
    mutate(temporada = temporada + 1L)

  df %>%
    left_join(anterior %>% rename(equipo_local = equipo, pct_victorias_anterior_local = pct),
              by = c("temporada", "equipo_local")) %>%
    left_join(anterior %>% rename(equipo_visitante = equipo, pct_victorias_anterior_visita = pct),
              by = c("temporada", "equipo_visitante"))
}

# Fuerza previa de cada equipo: cómo viene en la fase regular de la temporada,
# con los partidos anteriores al que se predice. Los de playoffs no cuentan: en
# un partido de playoffs, la fuerza es la de toda la fase regular.
#   fuerza_margen    : por cuánto gana en promedio (margen medio)
#   fuerza_victorias : qué proporción de partidos gana
# Arranca desde la fase regular de la temporada anterior, que vale
# PARTIDOS_ANTERIOR partidos, así que al principio pesa la anterior y después
# la actual:
#   fuerza = (suma de los partidos previos + 17 · valor de la anterior) / (partidos previos + 17)
# Un recién ascendido arranca con 0,35 de victorias y −5 de margen (los cinco
# del train promediaron 0,335 y −4,65). Sin descontar la localía: con el
# calendario casi alternado, no cambia nada.
PARTIDOS_ANTERIOR <- 17
ASCENDIDO_VICTORIAS <- 0.35
ASCENDIDO_MARGEN <- -5

agregar_fuerza <- function(df) {
  largo <- bind_rows(
    df %>% transmute(id_match, fecha, temporada, es_playoff, rol = "local",
                     equipo = equipo_local, gano = victoria_local,
                     margen = dif_puntos_final),
    df %>% transmute(id_match, fecha, temporada, es_playoff, rol = "visita",
                     equipo = equipo_visitante, gano = 1L - victoria_local,
                     margen = -dif_puntos_final)
  )

  anterior <- largo %>%
    filter(es_playoff == 0) %>%
    group_by(equipo, temporada) %>%
    summarise(victorias_ant = mean(gano), margen_ant = mean(margen), .groups = "drop") %>%
    bind_rows(RECORD_2020 %>% transmute(
      equipo, temporada = 2020L,
      victorias_ant = victorias / (victorias + derrotas),
      margen_ant    = (puntos_favor - puntos_contra) / (victorias + derrotas))) %>%
    mutate(temporada = temporada + 1L)

  fuerza <- largo %>%
    left_join(anterior, by = c("equipo", "temporada")) %>%
    mutate(victorias_ant = coalesce(victorias_ant, ASCENDIDO_VICTORIAS),
           margen_ant    = coalesce(margen_ant, ASCENDIDO_MARGEN),
           regular       = as.integer(es_playoff == 0)) %>%
    arrange(equipo, fecha, id_match) %>%
    group_by(equipo, temporada) %>%
    mutate(
      previos   = cumsum(regular) - regular,
      victorias = cumsum(gano * regular) - gano * regular,
      margenes  = cumsum(margen * regular) - margen * regular,
      fuerza_victorias = (victorias + PARTIDOS_ANTERIOR * victorias_ant) /
        (previos + PARTIDOS_ANTERIOR),
      fuerza_margen    = (margenes + PARTIDOS_ANTERIOR * margen_ant) /
        (previos + PARTIDOS_ANTERIOR)
    ) %>%
    ungroup() %>%
    select(id_match, rol, fuerza_margen, fuerza_victorias) %>%
    pivot_wider(names_from = rol, values_from = c(fuerza_margen, fuerza_victorias),
                names_glue = "{.value}_{rol}", names_sort = TRUE)

  df %>% left_join(fuerza, by = "id_match")
}

# El box score del corte, con las variables derivadas.
cargar_box_score <- function(corte) {
  ruta <- file.path("datos", "box_scores",
                    sprintf("box_score_%dc_%s.csv", corte, ETIQUETA_TEMPORADAS))
  # fileEncoding (no encoding): "Barça" y "Breogán" llegan bien en Windows y en
  # Linux, sin depender del locale.
  df <- utils::read.csv(ruta, stringsAsFactors = FALSE, fileEncoding = "UTF-8")

  df <- df %>%
    mutate(
      fecha = as.Date(fecha),
      # Temporada = año de inicio (la liga arranca en septiembre)
      temporada = as.integer(format(fecha, "%Y")) - (as.integer(format(fecha, "%m")) < 7),
      equipo_local     = canonizar_equipo(equipo_local),
      equipo_visitante = canonizar_equipo(equipo_visitante),
      # jornada: "1".."34" en fase regular; "Cuartos (1º)", "Final (2º)", etc.
      # en playoffs
      es_playoff = as.integer(!grepl("^[0-9]+$", trimws(jornada))),
      # Marcador al corte: dif_puntos_Q<corte> (la ventaja del local) ya viene
      # en el box score; total_puntos_Q<corte> es el ritmo (un 55-45 no es un
      # 35-25).
      "total_puntos_Q{corte}" := .data[[paste0("marcador_local_Q", corte)]] +
        .data[[paste0("marcador_visit_Q", corte)]]
    )

  # Cuatro factores de Dean Oliver (Basketball on Paper, 2004), de cada equipo y
  # acumulados hasta el corte. Miden cómo termina cada posesión, como tasa por
  # oportunidad, así que no dependen del ritmo. En escala 0-100: los OR de una
  # logística son por punto porcentual.
  #   efg : tiro de campo efectivo, (T2A + 1,5·T3A) / (T2I + T3I)
  #   tov : pérdidas por posesión, perdida / (T2I + T3I + 0,44·T1I + perdida)
  #   orb : rebote ofensivo, reb_of / (reb_of + reb_def del rival)
  #   ftr : tiros libres anotados por tiro de campo, T1A / (T2I + T3I)
  # Ojo: efg es casi el marcador (puntos = 2·(T2I + T3I)·efg/100 + T1A).
  df <- df %>%
    mutate(
      local_efg  = 100 * (local_T2A + 1.5 * local_T3A) / (local_T2I + local_T3I),
      local_tov  = 100 * local_perdida / (local_T2I + local_T3I + 0.44 * local_T1I + local_perdida),
      local_orb  = 100 * local_reb_of / (local_reb_of + visita_reb_def),
      local_ftr  = 100 * local_T1A / (local_T2I + local_T3I),
      visita_efg = 100 * (visita_T2A + 1.5 * visita_T3A) / (visita_T2I + visita_T3I),
      visita_tov = 100 * visita_perdida / (visita_T2I + visita_T3I + 0.44 * visita_T1I + visita_perdida),
      visita_orb = 100 * visita_reb_of / (visita_reb_of + local_reb_def),
      visita_ftr = 100 * visita_T1A / (visita_T2I + visita_T3I),
      corte_cuarto = corte
    )

  agregar_fuerza(agregar_temporada_anterior(agregar_partidos_jugados(df)))
}

# La matriz de features: todo lo de cargar_box_score() menos lo excluido. Qué
# subconjunto usa cada modelo se decide en su notebook. Del marcador quedan la
# diferencia al final de cada cuarto hasta el corte (dif_puntos_Q1, ...,
# dif_puntos_Q<corte>) y total_puntos_Q<corte>: los marcadores de cada equipo
# se reconstruirían desde la diferencia y el total.
construir_features <- function(df) {
  corte <- df$corte_cuarto[1]
  excluidas <- c(COLS_EXCLUIDAS,
                 paste0(c("marcador_local_Q", "marcador_visit_Q"), corte))
  df %>% select(-any_of(excluidas))
}
