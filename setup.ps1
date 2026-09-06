# Menyiapkan program ini di komputer baru.
# Jalankan sekali saja:  .\setup.ps1
#
# Yang dikerjakan:
#   1. mencari Python 3.11 atau 3.12
#   2. membuat folder .venv
#   3. memasang semua library dari requirements.txt
#   4. menguji hasilnya pakai gambar di folder contoh

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Judul($teks) {
    Write-Host ""
    Write-Host "=== $teks ===" -ForegroundColor Cyan
}

# --- 1. Cari cara membuat .venv -----------------------------------------------
Judul "Mencari Python 3.11 atau 3.12"

# MediaPipe belum menyediakan versi untuk Python 3.13 ke atas, jadi versi yang
# lebih baru sengaja tidak dipakai walaupun terpasang di komputer ini.
#
# Ada tiga kemungkinan cara, dicoba berurutan:
#   uv     - paling gampang, bisa mengunduh Python 3.12 sendiri kalau belum ada
#   py     - launcher bawaan Python di Windows
#   python - apa pun yang ada di PATH, asal versinya cocok
$cara = $null

if (Get-Command uv -ErrorAction SilentlyContinue) {
    $cara = "uv"
    Write-Host "Ketemu uv. Python 3.12 akan disiapkan otomatis." -ForegroundColor Green
}

if (-not $cara -and (Get-Command py -ErrorAction SilentlyContinue)) {
    foreach ($versi in @("3.12", "3.11")) {
        $hasil = & py "-$versi" --version 2>$null
        if ($LASTEXITCODE -eq 0) {
            $cara = "py"
            $pyVersi = "-$versi"
            Write-Host "Ketemu: $hasil" -ForegroundColor Green
            break
        }
    }
}

if (-not $cara -and (Get-Command python -ErrorAction SilentlyContinue)) {
    $versiPath = & python --version 2>$null
    if ($LASTEXITCODE -eq 0 -and $versiPath -match "3\.(11|12)\.") {
        $cara = "python"
        Write-Host "Ketemu: $versiPath" -ForegroundColor Green
    }
}

if (-not $cara) {
    Write-Host "Python 3.11 atau 3.12 tidak ditemukan." -ForegroundColor Red
    Write-Host ""
    Write-Host "Cara paling gampang, pasang uv dulu (nanti Python-nya diurus sendiri):"
    Write-Host "  winget install astral-sh.uv"
    Write-Host ""
    Write-Host "Atau pasang Python-nya langsung:"
    Write-Host "  winget install Python.Python.3.12"
    Write-Host "  atau unduh dari https://www.python.org/downloads/"
    Write-Host ""
    Write-Host "Catatan: Python 3.13 atau yang lebih baru tetap tidak bisa dipakai,"
    Write-Host "karena MediaPipe belum mendukungnya."
    exit 1
}

# --- 2. Buat .venv ------------------------------------------------------------
Judul "Menyiapkan folder .venv"

if (Test-Path ".venv\Scripts\python.exe") {
    Write-Host ".venv sudah ada, dipakai ulang. Hapus foldernya kalau mau mulai bersih."
} else {
    switch ($cara) {
        "uv"     { & uv venv --python 3.12 .venv }
        "py"     { & py $pyVersi -m venv .venv }
        "python" { & python -m venv .venv }
    }
    if ($LASTEXITCODE -ne 0) { Write-Host "Gagal membuat .venv" -ForegroundColor Red; exit 1 }
    Write-Host ".venv dibuat." -ForegroundColor Green
}

$pyVenv = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $pyVenv)) {
    Write-Host "python.exe tidak ada di dalam .venv" -ForegroundColor Red
    exit 1
}

# --- 3. Pasang library --------------------------------------------------------
Judul "Memasang library (paling lama di langkah ini)"
Write-Host "Ukuran unduhannya sekitar 600 MB, sabar ya."

if ($cara -eq "uv") {
    # uv jauh lebih cepat, dan .venv buatan uv memang belum berisi pip.
    & uv pip install --python $pyVenv -r requirements.txt
} else {
    & $pyVenv -m pip install --upgrade pip --quiet
    & $pyVenv -m pip install -r requirements.txt
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "Pemasangan library gagal." -ForegroundColor Red
    exit 1
}
Write-Host "Library terpasang." -ForegroundColor Green

# --- 4. Uji coba --------------------------------------------------------------
Judul "Menguji hasil pemasangan"
Write-Host "Menebak 4 gambar contoh, seharusnya semuanya benar."
Write-Host ""

& $pyVenv uji_gambar.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Uji coba gagal. Periksa pesan error di atas." -ForegroundColor Red
    exit 1
}

Judul "Selesai"
Write-Host "Program siap dipakai. Jalankan dengan:" -ForegroundColor Green
Write-Host "  .\run.ps1"
Write-Host ""
Write-Host "Atau di VS Code: tekan F5, lalu pilih 'Jalankan program (pakai kamera)'."
