$condaExe = Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"
(& $condaExe "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate PathViT-ray
Set-Location "D:\wonse\Documents\Job\urban_datalab\PATH-ViT"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
Remove-Item Env:\SSL_CERT_FILE -ErrorAction SilentlyContinue

# M1(순수 WSI, ABMIL)을 UNI 백본으로 5-fold external 검증. M3/PMA(UNI)와 같은 조건으로 맞추기
# 위해 --backbone uni만 바꾸고 AUG는 빼고 돌린다 - PMA/M3 양쪽에서 UNI+real-time augmentation이
# external을 오히려 떨어뜨리는 게 두 번 확인됐다(PMA: no-aug 0.644 -> strong-blur 0.614 ->
# weak-blur 0.602). data/patches_{tcga,cptac}/tiles/*/features_uni.pt가 이미 추출돼 있어
# --image 없이 precomputed 모드로 빠르게 돈다. M1/M2는 PMA의 비교군일 뿐이라 성능을 짜낼
# 필요는 없고, external c-index가 0.5를 넘는지만 확인하면 된다.
#
# 집계: python scripts/summarize_kfold.py --dataset tcga --seed 84 --n-folds 5 --model M1_uni_SS_DISP

$LogDir = ".logs"
$Seed = 84
$NFolds = 5
$GroupTs = "0807m1_uni_kfold5_local_seed84"

for ($fold = 0; $fold -lt $NFolds; $fold++) {
    Write-Host "=== M1_uni_SS_DISP fold=$fold/$NFolds Start: $(Get-Date) ==="
    $log = Join-Path $LogDir "train_tcga_seed${Seed}_M1_uni_SS_DISP_kfold5_fold${fold}.log"
    python -u .\train.py --M1 --dataset tcga --external --seed $Seed `
        --backbone uni `
        --patch-keep-frac 0.8 --attn-dispersion `
        --fold $fold --n-folds $NFolds --group-ts $GroupTs 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: M1_uni_SS_DISP fold=$fold" }
    Write-Host "=== M1_uni_SS_DISP fold=$fold/$NFolds Complete: $(Get-Date) ==="
}
Write-Host "=== ALL M1_uni_SS_DISP LOCAL 5-FOLD RUNS COMPLETE: $(Get-Date) ==="
