# Windows PowerShell version of convert.sh
$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path "pages" | Out-Null
Get-ChildItem -Path "pages" -Filter "*.html" -ErrorAction SilentlyContinue | Remove-Item -Force
Remove-Item -Path "pages/manifest.json" -ErrorAction SilentlyContinue

$nbs = Get-ChildItem -Path "notebooks" -Filter "*.ipynb" -ErrorAction SilentlyContinue
if (-not $nbs) {
    Write-Error "No .ipynb files in notebooks/"
    exit 1
}

$entries = @()
foreach ($nb in $nbs) {
    $base = $nb.BaseName
    jupyter nbconvert --to html --template basic --output-dir pages $nb.FullName
    $title = $base -replace '[_-]', ' '
    $entries += [pscustomobject]@{ title = $title; file = "$base.html" }
}

$entries | ConvertTo-Json -Compress | Set-Content -Path "pages/manifest.json" -Encoding utf8
Write-Host "Done. Manifest:"
Get-Content "pages/manifest.json"
