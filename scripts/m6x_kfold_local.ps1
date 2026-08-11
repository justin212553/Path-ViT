$condaExe = Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"
(& $condaExe "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate PathViT-ray
Set-Location "D:\wonse\Documents\Job\urban_datalab\PATH-ViT"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
Remove-Item Env:\SSL_CERT_FILE -ErrorAction SilentlyContinue

# M6X(RNAOnlyExtend, RNA 인코더 폭 G->256->256로 확장)를 M6(RNAOnly)와 동일 유전자셋(INT1500)으로
# 1시드(84, 프로젝트 기본값) 5-fold 파일럿. M6(RNA 단독, 좁은 인코더)가 지금까지 internal
# 1위(0.659)를 기록해서, RNA 브랜치 자체를 키우면 internal이 더 오르는지 확인하는 용도 —
# 신호가 있으면 유전자셋 크기(1000/2000)도 이어서 탐색.
#
# 집계: python scripts/pool_kfold_preds.py --dataset tcga --model M6X_INT1500 --seed 84 --n-folds 5

$LogDir = ".logs"
$Seed = 84
$NFolds = 5
$GroupTs = "0809m6x_int1500_kfold5_local_seed84"

for ($fold = 0; $fold -lt $NFolds; $fold++) {
    Write-Host "=== M6X_INT1500 fold=$fold/$NFolds Start: $(Get-Date) ==="
    $log = Join-Path $LogDir "train_tcga_seed${Seed}_M6X_INT1500_kfold5_fold${fold}.log"
    python -u .\train_light.py --M6X --rna-genes literature_1500_intersection --dataset tcga --external --seed $Seed `
        --fold $fold --n-folds $NFolds --group-ts $GroupTs 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: M6X_INT1500 fold=$fold" }
    Write-Host "=== M6X_INT1500 fold=$fold/$NFolds Complete: $(Get-Date) ==="
}
Write-Host "=== ALL M6X_INT1500 LOCAL 5-FOLD RUNS COMPLETE: $(Get-Date) ==="
