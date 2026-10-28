# ==============================================================================
# capturar.py — Cuotas de BetWarrior y Jugadón en los cortes de los partidos
# ==============================================================================
# Cuatro capturas por partido:
#
#   inicio     diez minutos antes de la hora, o al saltar si empieza antes
#   fin_2C     al terminar el 2do cuarto, ya en el entretiempo (reloj parado en 20:00)
#   fin_HT     13 minutos después, poco antes del 3er cuarto
#   fin_3C     en la pausa entre el 3er y el 4to cuarto (reloj parado en 30:00)
#
# De cada una se guardan ganador, hándicap y total, con prórroga incluida y con
# las líneas que publique cada casa.
#
# BetWarrior corre sobre Kambi, que publica la oferta, el reloj y el marcador en
# JSON sin login; su reloj es el que marca los momentos de captura. Jugadón
# (proveedor SRP/lsport) sólo responde a los pedidos que salen de su página, así
# que se la abre en un navegador sin ventana, como cualquier visitante, y se lee
# lo que la API le responde a la página: la vista del torneo, que trae los
# partidos que no empezaron, para emparejarlos con los de BetWarrior apenas
# aparecen en el calendario y para la captura de inicio, y la del partido en
# vivo para los cortes. (bplay se descartó: rechaza los navegadores automáticos.)
#
# Captura los partidos del día y termina; cuotas/programar.ps1 lo lanza todas
# las mañanas. Entre una captura y la siguiente la notebook se puede suspender:
# un temporizador de Windows la despierta a tiempo. El inicio tiene hora fija, y
# para los cortes se usa el reloj del partido: el reloj de juego nunca corre más
# rápido que el real, así que si faltan 8 minutos de juego para el corte, se
# puede volver a mirar en 8. Si una captura falla, o el mercado de
# ganador está suspendido, se reintenta en ESPERA segundos; si el ganador sigue
# suspendido después de MAX_SUSPENDIDO (pasa en las palizas), se guarda igual lo
# que haya, con abierta = False en los mercados suspendidos.
#
#   uv run python cuotas/capturar.py                                     # Liga ACB
#   uv run python cuotas/capturar.py basketball/euroleague/all/all 1234  # otra liga y
#                                                                        # su torneo en Jugadón
#
# Agrega filas a datos/cuotas/betwarrior.csv y datos/cuotas/jugadon.csv:
# cuota_1 es la del local (o "más de", en el total) y cuota_2 la del visitante
# (o "menos de"), y en el hándicap linea es la del local. En Jugadón id_evento es
# el de BetWarrior, para cruzar las dos casas, y desde es el marcador desde el
# que se cuenta el hándicap en vivo.
# ==============================================================================

import json
import socket
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
from playwright.sync_api import sync_playwright

RAIZ = Path(__file__).resolve().parents[1]
ZONA = "America/Argentina/Buenos_Aires"

# Liga de Kambi y torneo de Jugadón ("Liga ACB Endesa" es el 3111)
LIGA, TORNEO_JUGADON = sys.argv[1:3] if len(sys.argv) > 2 else ("basketball/spain/liga_acb/all", "3111")

KAMBI = "https://eu.offering-api.kambicdn.com/offering/v2018/bwargsf"
PARAMETROS = "?channel_id=1&client_id=200&lang=es_AR&market=AR"
MERCADOS = {1001159732: "ganador", 1001159512: "handicap", 1001159509: "total"}

JUGADON = ("https://santafe.jugadon.bet.ar/console.html"
           "?lang=es_AR&platform=191&css=santafe&play=true&sportId=48242")
MERCADOS_JUGADON = {"226": "ganador", "342": "handicap", "28": "total"}
MISMA_HORA = pd.Timedelta(minutes=30)  # para emparejar partidos entre casas
EN_JUGADON = {}  # id de BetWarrior -> id de Jugadón, se arma al aparecer

ANTES_DEL_INICIO = pd.Timedelta(minutes=10)  # hay partidos que empiezan antes de hora
FIN_DEL_ENTRETIEMPO = pd.Timedelta(minutes=13)  # el entretiempo dura 15
# Cota inferior de minutos reales por minuto de juego. Con 1,5 el script llegaba
# tarde a las pausas y perdía el fin_HT y el fin_3C (jornada 4, 17-18/10)
MINUTOS_REALES = 1
ESPERA = pd.Timedelta(seconds=30)
MAX_SUSPENDIDO = pd.Timedelta(seconds=30)  # un reintento: la pausa del 3C dura 2 minutos
SUSPENDER_DESDE = pd.Timedelta(minutes=10)
DURA_HASTA_3C = pd.Timedelta(minutes=105)  # del inicio al final del 3er cuarto, con margen

# Windows: pedirle al sistema que no se suspenda mientras se captura y que no
# congele el script (en espera moderna, con la tapa cerrada, Windows congela los
# programas aunque el sistema esté despierto), y despertar la notebook con un
# temporizador del sistema
if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    class MOTIVO(ctypes.Structure):  # REASON_CONTEXT con un texto simple
        _fields_ = [("Version", wintypes.ULONG), ("Flags", wintypes.DWORD),
                    ("Texto", wintypes.LPWSTR)]

    kernel32 = ctypes.WinDLL("kernel32")
    kernel32.PowerCreateRequest.argtypes = [ctypes.POINTER(MOTIVO)]
    kernel32.PowerCreateRequest.restype = wintypes.HANDLE
    kernel32.PowerSetRequest.argtypes = [wintypes.HANDLE, ctypes.c_int]
    kernel32.PowerClearRequest.argtypes = [wintypes.HANDLE, ctypes.c_int]
    kernel32.CreateWaitableTimerW.restype = wintypes.HANDLE
    kernel32.SetWaitableTimer.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_longlong),
                                          wintypes.LONG, ctypes.c_void_p, ctypes.c_void_p,
                                          wintypes.BOOL]
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    SISTEMA = 1  # PowerRequestSystemRequired: que no se suspenda
    EJECUCION = 3  # PowerRequestExecutionRequired: que no congele el proceso
    PEDIDO = kernel32.PowerCreateRequest(ctypes.byref(MOTIVO(0, 1, "Captura de cuotas de la tesina")))


def despierta(si):
    """Pide (o deja de pedir) que la notebook siga despierta y el script corriendo."""
    if sys.platform == "win32":
        for pedido in (SISTEMA, EJECUCION):
            (kernel32.PowerSetRequest if si else kernel32.PowerClearRequest)(PEDIDO, pedido)


def pedir(ruta):
    with urlopen(KAMBI + ruta + PARAMETROS, timeout=15) as respuesta:
        return json.load(respuesta)


def hora():
    return f"{pd.Timestamp.now(tz=ZONA):%Y-%m-%d %H:%M:%S}"


def nombre(partido):
    return f"{partido['event']['homeName']} - {partido['event']['awayName']}"


def milesimas(x):
    return x / 1000 if x is not None else None


def que_capturar(partido, ahora, inicio_pausa):
    """Qué captura le toca al partido en este momento, o None."""
    evento = partido["event"]
    if evento["state"] == "NOT_STARTED":
        return "inicio" if pd.Timestamp(evento["start"]) - ahora <= ANTES_DEL_INICIO else None
    reloj = partido.get("liveData", {}).get("matchClock")
    if reloj is None:  # sin reloj no se sabe en qué momento del partido está
        return None
    segundos = 60 * reloj.get("minute", 0) + reloj.get("second", 0)
    if segundos == 0:
        return "inicio"
    if reloj.get("running", True):
        return None
    if segundos == 1800:
        return "fin_3C"
    if segundos == 1200:
        inicio = inicio_pausa.setdefault(evento["id"], ahora)
        return "fin_HT" if ahora - inicio >= FIN_DEL_ENTRETIEMPO else "fin_2C"
    return None


def proxima_mirada(partido, ahora, inicio_pausa):
    """Hasta cuándo se puede esperar sin perder una captura del partido, o None
    si ya no le queda ninguna."""
    evento = partido["event"]
    if evento["state"] == "NOT_STARTED":
        return pd.Timestamp(evento["start"]) - ANTES_DEL_INICIO
    reloj = partido.get("liveData", {}).get("matchClock", {})
    segundos = 60 * reloj.get("minute", 0) + reloj.get("second", 0)
    if segundos > 1800:
        return None
    if segundos == 0:
        return ahora
    if segundos == 1200 and evento["id"] in inicio_pausa:
        return inicio_pausa[evento["id"]] + FIN_DEL_ENTRETIEMPO
    # Con el reloj en 20:00 todavía corriendo, el entretiempo está por empezar
    corte = 1200 if segundos <= 1200 else 1800
    return ahora + pd.Timedelta(minutes=MINUTOS_REALES * (corte - segundos) / 60)


def dormir_hasta(momento):
    """Espera hasta momento. En Windows espera un temporizador que despierta la
    notebook si se suspendió en el medio (por ejemplo, al cerrar la tapa), y si
    falta mucho la deja suspenderse."""
    falta = momento - pd.Timestamp.now(tz="UTC")
    if sys.platform != "win32":
        time.sleep(max(0, falta.total_seconds()))
        return
    temporizador = kernel32.CreateWaitableTimerW(None, True, None)
    # Hora absoluta, en intervalos de 100 ns desde 1601; el último True despierta al sistema
    vence = ctypes.c_longlong(int((momento.timestamp() + 11_644_473_600) * 10**7))
    kernel32.SetWaitableTimer(temporizador, ctypes.byref(vence), 0, None, None, True)
    if falta >= SUSPENDER_DESDE:
        despierta(False)
    kernel32.WaitForSingleObject(temporizador, 0xFFFFFFFF)
    despierta(True)
    kernel32.CloseHandle(temporizador)


def guardar(filas, casa):
    archivo = RAIZ / "datos" / "cuotas" / f"{casa}.csv"
    pd.DataFrame(filas).to_csv(archivo, mode="a", header=not archivo.exists(), index=False)


def capturar_betwarrior(partido, captura):
    """Las cuotas de BetWarrior del partido: una fila por línea de cada mercado."""
    evento = partido["event"]
    marcador = partido.get("liveData", {}).get("score", {})
    comunes = {
        "momento": pd.Timestamp.now(tz="UTC").floor("s"),
        "captura": captura,
        "id_evento": evento["id"],
        "local": evento["homeName"],
        "visitante": evento["awayName"],
        "inicio": evento["start"],
        "puntos_local": marcador.get("home"),
        "puntos_visita": marcador.get("away"),
    }
    filas = []
    for apuesta in pedir(f"/betoffer/event/{evento['id']}.json")["betOffers"]:
        if apuesta["criterion"]["id"] not in MERCADOS:
            continue
        uno = next(o for o in apuesta["outcomes"] if o["type"] in ("OT_ONE", "OT_OVER"))
        dos = next(o for o in apuesta["outcomes"] if o["type"] in ("OT_TWO", "OT_UNDER"))
        filas.append(comunes | {
            "mercado": MERCADOS[apuesta["criterion"]["id"]],
            "linea": milesimas(uno.get("line")),
            "cuota_1": milesimas(uno.get("odds")),
            "cuota_2": milesimas(dos.get("odds")),
            "principal": bool({"MAIN", "MAIN_LINE"} & set(apuesta.get("tags", []))),
            "abierta": not apuesta.get("suspended", False)
                       and uno["status"] == dos["status"] == "OPEN",
        })
    return filas


def leer_jugadon(vista, servicio):
    """Abre una vista de la página de Jugadón en un navegador sin ventana y
    devuelve lo que la API le responde a la página en servicio."""
    with sync_playwright() as p:
        pagina = p.chromium.launch().new_page()
        with pagina.expect_response(lambda r: servicio in r.url, timeout=60_000) as respuesta:
            pagina.goto(JUGADON + vista)
        return respuesta.value.json()


def emparejar(evento, candidatos):
    """El partido de Jugadón de un evento de BetWarrior: el que empieza a la misma
    hora con los nombres más parecidos (no coinciden entre casas: "La Laguna
    Tenerife" en una es "CB 1939 Canarias" en la otra)."""
    inicio = pd.Timestamp(evento["start"])
    cerca = [c for c in candidatos if abs(pd.Timestamp(c["scheduled"]) - inicio) <= MISMA_HORA]
    if not cerca:
        raise ValueError("no está en la vista del torneo de Jugadón")

    def parecido(a, b):
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    return max(cerca, key=lambda c: parecido(evento["homeName"], c["home"])
                                    + parecido(evento["awayName"], c["away"]))


def emparejar_jugadon(partidos):
    """Empareja con Jugadón los partidos que todavía no empezaron, apenas
    aparecen en el calendario. Los cortes necesitan el partido emparejado, y si
    se esperaba a la captura de inicio, perderla dejaba sin Jugadón el partido
    entero (el 2026-10-14 pasó con Joventut - Andorra). Si falla, se vuelve a
    intentar en la captura de inicio."""
    partidos = [p for p in partidos if p["event"]["state"] == "NOT_STARTED"]
    if not partidos:
        return
    try:
        datos = leer_jugadon(f"&tournamentId={TORNEO_JUGADON}", "prematch/tournament/preview")
    except Exception as error:
        print(f"{hora()}  jugadon: error al leer el torneo: "
              f"{(str(error) or repr(error)).splitlines()[0]}")
        return
    candidatos = [e for dia in datos["events"] for e in dia["events"]]
    for p in partidos:
        try:
            partido_j = emparejar(p["event"], candidatos)
        except ValueError as error:
            print(f"{hora()}  {nombre(p)}, jugadon: {error}")
            continue
        EN_JUGADON[p["event"]["id"]] = partido_j["id"]
        print(f"{hora()}  {nombre(p)}, jugadon: emparejado con "
              f"{partido_j['home']} - {partido_j['away']}")


def linea_jugadon(linea):
    """"-5.5 (39-39)" -> (-5.5, "39-39"): en vivo el hándicap se cuenta desde ese marcador."""
    if linea is None:
        return None, None
    numero, _, desde = str(linea).partition(" ")
    return float(numero), desde.strip("()") or None


def capturar_jugadon(partido, captura):
    """Las cuotas de Jugadón del partido: una fila por línea de cada mercado."""
    evento = partido["event"]
    if evento["state"] == "NOT_STARTED":
        datos = leer_jugadon(f"&tournamentId={TORNEO_JUGADON}", "prematch/tournament/preview")
        partido_j = emparejar(evento, [e for dia in datos["events"] for e in dia["events"]])
        EN_JUGADON[evento["id"]] = partido_j["id"]
    else:
        if evento["id"] not in EN_JUGADON:
            raise ValueError("no se emparejó, ni al aparecer ni en el inicio")
        datos = leer_jugadon(f"&openlivesection=true&tournamentId={TORNEO_JUGADON}"
                             f"&eventId={EN_JUGADON[evento['id']]}", "live/event-detail")
        partido_j = datos["event"]
    marcador = {r["position"]: r["value"] for r in
                (partido_j.get("livescore") or {}).get("scoreboard", {}).get("results", [])}
    comunes = {
        "momento": pd.Timestamp.now(tz="UTC").floor("s"),
        "captura": captura,
        "id_evento": evento["id"],
        "id_jugadon": partido_j["id"],
        "local": partido_j["home"],
        "visitante": partido_j["away"],
        "inicio": partido_j["scheduled"],
        "puntos_local": marcador.get("1"),
        "puntos_visita": marcador.get("2"),
    }
    filas = []
    # En la vista del partido los mercados vienen agrupados en listas
    for grupo in partido_j.get("odds") or []:
        for mercado in grupo if isinstance(grupo, list) else [grupo]:
            if mercado["id"] not in MERCADOS_JUGADON:
                continue
            uno, dos = sorted(mercado["outcome"], key=lambda o: o["sort"])
            linea, desde = linea_jugadon(uno.get("line"))
            filas.append(comunes | {
                "mercado": MERCADOS_JUGADON[mercado["id"]],
                "linea": linea,
                "desde": desde,
                "cuota_1": float(uno["outcome_odd_value"]) or None,
                "cuota_2": float(dos["outcome_odd_value"]) or None,
                "principal": uno.get("baseLine") == uno.get("mainLine"),
                "abierta": str(uno["status"]) == str(dos["status"]) == "1",
            })
    return filas


def main():
    # Una sola captura a la vez: si el puerto está tomado, ya hay otra corriendo
    candado = socket.socket()
    try:
        candado.bind(("127.0.0.1", 47321))
    except OSError:
        print(f"{hora()}  ya hay otra captura corriendo")
        return
    despierta(True)
    (RAIZ / "datos" / "cuotas").mkdir(parents=True, exist_ok=True)
    # BetWarrior primero: Jugadón abre un navegador y tarda más
    casas = {"betwarrior": capturar_betwarrior}
    if TORNEO_JUGADON:
        casas["jugadon"] = capturar_jugadon
    hoy = pd.Timestamp.now(tz=ZONA).date()
    capturados = set()
    primer_intento = {}
    inicio_pausa = {}
    vistos = {}  # los partidos del día que ya aparecieron en el calendario
    avisados = set()
    while True:
        ahora = pd.Timestamp.now(tz="UTC").floor("s")
        # Un corte de internet no tiene que terminar la captura de toda la jornada
        try:
            partidos = [p for p in pedir(f"/listView/{LIGA}/matches.json")["events"]
                        if p["event"]["state"] in ("NOT_STARTED", "STARTED")
                        and pd.Timestamp(p["event"]["start"]).tz_convert(ZONA).date() == hoy]
        except Exception as error:
            print(f"{hora()}  error al leer el calendario: {error}")
            dormir_hasta(ahora + ESPERA)
            continue
        # Kambi a veces saca del calendario un partido en juego (el 2026-10-10 pasó
        # con Breogán - Girona): hasta la hora probable de su 3C se lo sigue esperando
        en_lista = {p["event"]["id"] for p in partidos}
        faltan = [p for i, p in vistos.items() if i not in en_lista
                  and ahora < pd.Timestamp(p["event"]["start"]) + DURA_HASTA_3C]
        for p in faltan:
            if p["event"]["id"] not in avisados:
                print(f"{hora()}  {nombre(p)}: no está en el calendario de Kambi, se lo sigue esperando")
                avisados.add(p["event"]["id"])
        # Para diagnosticar inicios perdidos: cada partido al aparecer, y cada cambio
        # de su hora de inicio (Kambi la corrige al empezar el partido)
        for p in partidos:
            anterior = vistos.get(p["event"]["id"])
            if anterior is None or anterior["event"]["start"] != p["event"]["start"]:
                print(f"{hora()}  en el calendario: {nombre(p)}, "
                      f"{pd.Timestamp(p['event']['start']).tz_convert(ZONA):%d/%m %H:%M}")
        nuevos = [p for p in partidos if p["event"]["id"] not in vistos]
        vistos.update({p["event"]["id"]: p for p in partidos})
        if not partidos and not faltan:
            print(f"{hora()}  no quedan partidos de {LIGA} hoy")
            return
        if "jugadon" in casas and nuevos:
            emparejar_jugadon(nuevos)

        a_capturar = [(p, c) for p in partidos if (c := que_capturar(p, ahora, inicio_pausa))]
        pendiente = False
        for casa, capturar in casas.items():
            for p, captura in a_capturar:
                clave = (casa, p["event"]["id"], captura)
                if clave in capturados:
                    continue
                try:
                    filas = capturar(p, captura)
                except Exception as error:
                    # Los errores de Playwright traen el registro de llamadas abajo
                    print(f"{hora()}  {nombre(p)}, {captura}, {casa}: error: "
                          f"{(str(error) or repr(error)).splitlines()[0]}")
                    pendiente = True
                    continue
                abierto = any(f["mercado"] == "ganador" and f["abierta"] for f in filas)
                if abierto or (filas and ahora - primer_intento.setdefault(clave, ahora) >= MAX_SUSPENDIDO):
                    guardar(filas, casa)
                    capturados.add(clave)
                    estado = "guardada" if abierto else "guardada con el ganador suspendido"
                else:
                    pendiente = True
                    estado = "ganador suspendido, se reintenta"
                print(f"{hora()}  {nombre(p)}, {captura}, {casa}: {estado}")

        miradas = [m for p in partidos if (m := proxima_mirada(p, ahora, inicio_pausa)) is not None]
        if faltan:
            miradas.append(ahora + ESPERA)
        if not miradas:
            print(f"{hora()}  no quedan capturas hoy")
            return
        hasta = ahora + ESPERA if pendiente else max(min(miradas), ahora + ESPERA)
        if hasta - ahora >= SUSPENDER_DESDE:
            print(f"{hora()}  se vuelve a mirar a las {hasta.tz_convert(ZONA):%H:%M}")
        dormir_hasta(hasta)

if __name__ == "__main__":
    main()
