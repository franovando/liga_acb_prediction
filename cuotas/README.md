# Cuotas en vivo (Python)

Las cuotas en los cortes no se consiguen después del partido, así que se capturan en el momento. Esto corre aparte del pipeline, en una notebook con Windows y sin intervención: una tarea programada lanza el script cada mañana, y el script despierta la notebook para cada captura y la deja suspendida el resto del tiempo.

Está hecho y probado en Windows, con Python (vía uv), Playwright con Chromium para leer Jugadón, el Programador de tareas de Windows (lo configura `programar.ps1`) y los temporizadores de reactivación de Windows para despertar la notebook. Los comandos de este README son para PowerShell y se corren desde la raíz del repo.

## capturar.py · BetWarrior y Jugadón

Guarda las cuotas de ganador, hándicap y total (con prórroga incluida y todas sus líneas) de BetWarrior y Jugadón, en cuatro momentos de cada partido de la ACB:

- `inicio`: diez minutos antes de la hora, porque hay partidos que empiezan antes;
- `fin_2C`: al terminar el 2° cuarto;
- `fin_HT`: 13 minutos después, poco antes del 3er cuarto;
- `fin_3C`: en la pausa entre el 3er y el 4to cuarto.

El reloj sale de BetWarrior, que corre sobre Kambi y publica su oferta en JSON sin login. Jugadón sólo responde a los pedidos que salen de su página, así que el script la abre en un navegador sin ventana (Chromium, con Playwright). Si el mercado de ganador está suspendido, reintenta cada 30 segundos. Corre una sola captura a la vez, captura sólo los partidos del día y termina.

Entre una captura y la siguiente, deja que la notebook se suspenda y la despierta con un temporizador de Windows. El inicio tiene hora fija; para los cortes usa el reloj del partido, sabiendo que el reloj de juego nunca corre más rápido que el real. Nunca fuerza la suspensión: si la notebook se está usando, sigue prendida.

Antes de la primera captura hay que bajar el navegador:

```powershell
uv run playwright install chromium
```

Para correrlo a mano, con la Liga ACB:

```powershell
uv run python cuotas/capturar.py
```

Para probarlo con otra liga se le pasan dos argumentos, la liga en Kambi y el número del torneo en Jugadón (el de la Liga ACB es 3111). Por ejemplo, para la Euroliga: `uv run python cuotas/capturar.py basketball/euroleague/all/all ID_JUGADON`, con el número del torneo en lugar de `ID_JUGADON`.

## programar.ps1 · La tarea de Windows

Crea la tarea `tesina-cuotas`, que todos los días a las 6:30 despierta la notebook si está suspendida y corre `capturar.py`; si ese día no hay partidos, termina enseguida y Windows la vuelve a suspender. También corre al iniciar sesión, para retomar el día si la notebook se apagó en el medio (en ese caso puede repetirse una captura ya hecha).

La notebook tiene que quedar suspendida (no apagada, porque apagada Windows no la puede despertar), enchufada y con los temporizadores de reactivación permitidos. Todo esto se corre una sola vez, en una PowerShell abierta como administrador, desde la raíz del repo:

```powershell
# El botón de encendido y la tapa suspenden, enchufada y con batería
powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS PBUTTONACTION 1
powercfg /setdcvalueindex SCHEME_CURRENT SUB_BUTTONS PBUTTONACTION 1
powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 1
powercfg /setdcvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 1

# Los temporizadores pueden despertar la notebook. Con batería, gasta batería: cuidado en la mochila
powercfg /setacvalueindex SCHEME_CURRENT SUB_SLEEP RTCWAKE 1
powercfg /setdcvalueindex SCHEME_CURRENT SUB_SLEEP RTCWAKE 1
powercfg /setactive SCHEME_CURRENT

# La tarea
powershell -ExecutionPolicy Bypass -File cuotas\programar.ps1
```

## Archivos generados

| Archivo | Descripción |
| --- | --- |
| `datos/cuotas/betwarrior.csv` y `datos/cuotas/jugadon.csv` | Cuotas de cada casa: una fila por partido, captura, mercado y línea |
| `datos/cuotas/capturar.log` | Lo que imprime `capturar.py` cuando lo lanza la tarea de Windows |
