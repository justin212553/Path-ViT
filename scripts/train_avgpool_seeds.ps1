# seed 3개(42,84,126) x TCGA/CPTAC 순으로 train.py --avgpool을 직렬 실행 (PowerShell)
# ViT_M1의 ABMIL을 무학습 평균 풀링으로 대체한 ablation. internal test만 확인(--external 없음).
#
# 사용법:
#   .\scripts\train_avgpool_seeds.ps1

$ErrorActionPreference = "Continue"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir   = Split-Path -Parent $ScriptDir
$LogDir    = Join-Path $RootDir ".logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

Set-Location $RootDir

$condaExe = Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"
$needsActivate = -not (Get-Command python -ErrorAction SilentlyContinue) `
    -or ((python -c "import sys; print(sys.prefix)") -notmatch "PathViT-ray")
if ($needsActivate) {
    if (-not (Test-Path $condaExe)) {
        Write-Error "conda.exe를 찾을 수 없습니다($condaExe). PathViT-ray 환경을 먼저 activate하세요."
        exit 1
    }
    (& $condaExe "shell.powershell" "hook") | Out-String | Invoke-Expression
    conda activate PathViT-ray
    if (-not $?) {
        Write-Error "conda activate PathViT-ray 실패."
        exit 1
    }
}

$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$Seeds    = @(42, 84, 126)
$Datasets = @("tcga", "cptac")

$Total = $Seeds.Count * $Datasets.Count
$Run   = 0

foreach ($seed in $Seeds) {
    foreach ($dataset in $Datasets) {
        $Run++
        Write-Host "=== [$Run/$Total] avgpool dataset=$dataset seed=$seed Train Start: $(Get-Date) ==="
        $log = Join-Path $LogDir "train_${dataset}_seed${seed}_avgpool.log"
        python -u .\train.py --dataset $dataset --seed $seed --avgpool | Tee-Object -FilePath $log
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        Write-Host "=== [$Run/$Total] avgpool dataset=$dataset seed=$seed Train Complete: $(Get-Date) ==="
    }
}
