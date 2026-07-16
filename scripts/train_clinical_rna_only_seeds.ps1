# seed 3개(42,84,126) x TCGA/CPTAC 순으로 train_clinical_rna_only.py를 직렬 실행 (PowerShell)
# WSI 없이 age/sex+RNA만 쓰는 대조군 — ViT 복잡도 vs 표본 크기 노이즈를 구분하기 위한 실험.
#
# 사용법:
#   .\scripts\train_clinical_rna_only_seeds.ps1

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

$Seeds    = @(42, 84, 126)
$Datasets = @("tcga", "cptac")

$Total = $Seeds.Count * $Datasets.Count
$Run   = 0

foreach ($seed in $Seeds) {
    foreach ($dataset in $Datasets) {
        $Run++
        Write-Host "=== [$Run/$Total] clinical_rna_only dataset=$dataset seed=$seed Train Start: $(Get-Date) ==="
        $log = Join-Path $LogDir "train_${dataset}_seed${seed}_clinicalrnaonly.log"
        python -u .\train_clinical_rna_only.py --dataset $dataset --seed $seed | Tee-Object -FilePath $log
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        Write-Host "=== [$Run/$Total] clinical_rna_only dataset=$dataset seed=$seed Train Complete: $(Get-Date) ==="
    }
}
