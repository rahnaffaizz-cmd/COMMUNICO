# Menjalankan pengenal huruf BISINDO yang bisa bersuara (Google TTS).
#
# Menjalankan biasa
#   .\run.ps1                       suara bahasa Indonesia
#   .\run.ps1 --kamera 1            kalau kamera yang kebuka salah
#   .\run.ps1 --bahasa en --pelan   suara bahasa Inggris, dibaca pelan
#   .\run.ps1 --tanpa-eja-huruf     cuma kata yang dibunyikan, hurufnya tidak
#
# Kalau hurufnya susah masuk atau malah salah masuk
#   .\run.ps1 --min-keyakinan 0.6   lebih ketat, biar tidak gampang salah huruf
#   .\run.ps1 --ambang-seragam      abaikan ambang per huruf di ambang_huruf.json,
#                                   huruf yang biasanya ditolak jadi ikut masuk
#
# Kalau ingin tahu apa yang sebenarnya terjadi
#   .\run.ps1 --catat-detail        catat tiap tebakan beserta pesaingnya, dan
#                                   alasan kenapa sebuah huruf belum diterima
#   .\run.ps1 --ukur I              peragakan huruf I terus-menerus, nanti
#                                   dilaporkan seberapa baik huruf itu terbaca
#                                   oleh kamera Anda sendiri
#
# Menguji tanpa kamera
#   .\run.ps1 -Uji                  pakai gambar di folder contoh
#   .\run.ps1 -Uji "C:\foto\saya"   pakai gambar di folder lain
#
# Daftar lengkap:  .\run.ps1 -h
param(
    [switch]$Uji,
    [Parameter(ValueFromRemainingArguments = $true)] $Args
)

Set-Location $PSScriptRoot

# Cari Python-nya berurutan: .venv di folder ini, .venv proyek induk, lalu PATH.
$kandidat = @(
    (Join-Path $PSScriptRoot ".venv\Scripts\python.exe"),
    (Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe")
)
$py = $kandidat | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $py) {
    $py = (Get-Command python -ErrorAction SilentlyContinue).Source
}

if (-not $py) {
    Write-Host "Python tidak ditemukan." -ForegroundColor Red
    Write-Host "Siapkan dulu dengan perintah berikut di folder ini:"
    Write-Host "  uv venv --python 3.12 .venv"
    Write-Host "  uv pip install --python .venv\Scripts\python.exe -r requirements.txt"
    Write-Host "Kalau belum punya uv, bisa juga:"
    Write-Host "  py -3.12 -m venv .venv"
    Write-Host "  .venv\Scripts\pip install -r requirements.txt"
    exit 1
}

$skrip = if ($Uji) { "uji_gambar.py" } else { "gesture_voice.py" }
& $py $skrip @Args
