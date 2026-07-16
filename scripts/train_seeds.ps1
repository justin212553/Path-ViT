# seed 5개(42*1~5) x TCGA/CPTAC 순으로 train.py를 로컬 GPU 1장에서 직렬 실행 (PowerShell)
# internal test 성능이 seed에 따라 얼마나 흔들리는지(진짜 실력 vs 노이즈) 확인하기 위한 반복 실험용.
# --external을 항상 같이 켜서, seed마다 external(반대 코호트) 결과도 덤으로 확보한다.
#
# 사용법:
#   .\scripts\train_seeds.ps1                # 5 seed x (tcga, cptac) = 10회 순차 실행
#   .\scripts\train_seeds.ps1 --M2            # 나머지 인자는 모든 실행에 그대로 전달됨

$ErrorActionPreference = "Continue"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir   = Split-Path -Parent $ScriptDir
$LogDir    = Join-Path $RootDir ".logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

Set-Location $RootDir

# PathViT-ray 환경이 이미 activate돼 있지 않은 셸(예: 백그라운드/비대화 세션)에서도 안전하게
# 돌아가도록, python이 base 환경을 가리키고 있으면 conda hook을 초기화하고 직접 activate한다.
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

$Seeds    = 1..5 | ForEach-Object { 42 * $_ }   # 42, 84, 126, 168, 210
$Datasets = @("tcga", "cptac")

# --fusion/--M2/--M4 등은 로그 파일명에 접미사로 남겨 구분한다.
$Suffix = ""
if ($args -contains "--fusion") { $Suffix += "_fusion" }
if ($args -contains "--M2")     { $Suffix += "_M2" }
if ($args -contains "--M4")     { $Suffix += "_M4" }
if ($args -contains "--M4A")    { $Suffix += "_M4A" }

# [wandb Group] 이 스윕에서 나오는 모든 run(모든 시드/코호트, internal+external)이 하나의
# wandb Group(<모델종류>_<GroupTs>)으로 묶이도록, 첫 실행 전에 타임스탬프를 한 번만 계산해
# 모든 python train.py 호출에 동일하게 넘긴다(train.py --group-ts 참조).
$GroupTs = Get-Date -Format "MMdd::HHmm"

$Total = $Seeds.Count * $Datasets.Count
$Run   = 0

foreach ($seed in $Seeds) {
    foreach ($dataset in $Datasets) {
        $Run++
        Write-Host "=== [$Run/$Total] dataset=$dataset seed=$seed Train Start: $(Get-Date) ==="
        $log = Join-Path $LogDir "train_${dataset}_seed${seed}${Suffix}.log"
        python -u .\train.py --dataset $dataset --seed $seed --external --group-ts $GroupTs @args | Tee-Object -FilePath $log
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        Write-Host "=== [$Run/$Total] dataset=$dataset seed=$seed Train Complete: $(Get-Date) ==="
    }
}
