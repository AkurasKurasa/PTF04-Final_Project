# Launch Flask (port 8000) + JupyterLab (port 8888) in parallel.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

Write-Host "Starting JupyterLab on 8888..." -ForegroundColor Cyan
$lab = Start-Process -FilePath "jupyter" `
  -ArgumentList "lab","--no-browser","--port=8888","--ServerApp.token=''","--ServerApp.password=''","--ServerApp.disable_check_xsrf=True","--ServerApp.allow_origin='*'","--ServerApp.tornado_settings={`"headers`":{`"Content-Security-Policy`":`"frame-ancestors *`"}}","--notebook-dir=notebooks" `
  -PassThru -NoNewWindow

Write-Host "Starting Flask on 8000..." -ForegroundColor Cyan
$flask = Start-Process -FilePath "python" `
  -ArgumentList "backend/app.py" `
  -PassThru -NoNewWindow

Write-Host "`nSite: http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "JupyterLab: http://127.0.0.1:8888/lab" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop both." -ForegroundColor Yellow

try {
    Wait-Process -Id $lab.Id, $flask.Id
} finally {
    Stop-Process -Id $lab.Id, $flask.Id -Force -ErrorAction SilentlyContinue
}
