# ==============================================================================
# programar.ps1 — Tarea de Windows que lanza capturar.py todas las mañanas
# ==============================================================================
# Se corre una sola vez, en una PowerShell abierta como administrador, desde la
# raíz del repo:
#
#   powershell -ExecutionPolicy Bypass -File cuotas\programar.ps1
#
# Crea (o reemplaza) la tarea "tesina-cuotas": todos los días a las 6:30
# despierta la notebook si está suspendida y corre capturar.py, que captura los
# partidos de ese día (dejándola suspenderse entre captura y captura) y
# termina. Si a esa hora estaba apagada, corre apenas se prenda, y también corre
# al iniciar sesión, para retomar el día si la notebook se apagó en el medio.
# Corre sin ventana; lo que imprime queda en datos\cuotas\capturar.log.
# ==============================================================================

$repo = Split-Path -Parent $PSScriptRoot
$uv = (Get-Command uv).Source
New-Item -ItemType Directory -Force (Join-Path $repo "datos\cuotas") | Out-Null

$accion = New-ScheduledTaskAction -Execute "cmd.exe" -WorkingDirectory $repo `
    -Argument "/c `"`"$uv`" run python -X utf8 -u cuotas\capturar.py >> datos\cuotas\capturar.log 2>&1`""
$disparadores = @(
    New-ScheduledTaskTrigger -Daily -At "06:30"
    New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
)
$opciones = New-ScheduledTaskSettingsSet -WakeToRun -StartWhenAvailable `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 20)
$usuario = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType S4U

Register-ScheduledTask -TaskName "tesina-cuotas" -Action $accion -Trigger $disparadores `
    -Settings $opciones -Principal $usuario -Force
