# TCGA-PAAD -> CPTAC-PDA 순으로 train.py를 로컬 GPU 1장에서 직렬 실행 (PowerShell)
# (SLURM 없이 로컬 컴퓨터에서, conda 환경이 이미 activate된 PowerShell 세션에서 바로 실행하는 용도)
#
# 사용법:
#   .\scripts\train_serial.ps1                # tcga -> cptac baseline 순차 실행
#   .\scripts\train_serial.ps1 --fusion        # 나머지 인자는 두 실행에 그대로 전달됨

$ErrorActionPreference = "Continue"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir   = Split-Path -Parent $ScriptDir
$LogDir    = Join-Path $RootDir ".logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

Set-Location $RootDir

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "python을 찾을 수 없습니다. conda 환경(PathViT-ray)을 먼저 activate하세요: conda activate PathViT-ray"
    exit 1
}

$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

# --fusion이면 로그 파일명에 _fusion 접미사를 붙여 baseline 로그와 구분
$Suffix = ""
if ($args -contains "--fusion") {
    $Suffix = "_fusion"
}

Write-Host "=== [1/2] TCGA-PAAD Train Start: $(Get-Date) ==="
$tcgaLog = Join-Path $LogDir "train_tcga$Suffix.log"
python -u .\train.py --dataset tcga @args | Tee-Object -FilePath $tcgaLog
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "=== [1/2] TCGA-PAAD Train Complete: $(Get-Date) ==="

Write-Host "=== [2/2] CPTAC-PDA Train Start: $(Get-Date) ==="
$cptacLog = Join-Path $LogDir "train_cptac$Suffix.log"
python -u .\train.py --dataset cptac @args | Tee-Object -FilePath $cptacLog
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "=== [2/2] CPTAC-PDA Train Complete: $(Get-Date) ==="
