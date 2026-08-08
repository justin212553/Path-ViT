$condaExe = Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"
(& $condaExe "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate PathViT-ray
Set-Location "D:\wonse\Documents\Job\urban_datalab\PATH-ViT"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
Remove-Item Env:\SSL_CERT_FILE -ErrorAction SilentlyContinue

# M2_POOL(WSI MultiComponentPooling + Clinical(age/sex) co-attention query)을 UNI 백본으로
# 5-fold external 검증. models/vit_m2_pool.py::ViT_M2_Pool.combine_with_clinical_pool()이 이미
# PMA가 z_rna로 하는 것과 대칭으로 z_clinical을 co-attention query로 쓰도록 구현돼 있어(2026-08-07
# 요청 그대로) 코드 변경 없이 바로 사용. AUG는 UNI에서 손해라 빼고, precomputed features_uni.pt
# 사용(--image 불필요). 최소 기준은 external c>0.5.
#
# 집계: python scripts/summarize_kfold.py --dataset tcga --seed 84 --n-folds 5 --model M2_POOL_uni_SS_DISP

$LogDir = ".logs"
$Seed = 84
$NFolds = 5
$GroupTs = "0807m2pool_uni_kfold5_local_seed84"

for ($fold = 0; $fold -lt $NFolds; $fold++) {
    Write-Host "=== M2_POOL_uni_SS_DISP fold=$fold/$NFolds Start: $(Get-Date) ==="
    $log = Join-Path $LogDir "train_tcga_seed${Seed}_M2_POOL_uni_SS_DISP_kfold5_fold${fold}.log"
    python -u .\train.py --M2_POOL --dataset tcga --external --seed $Seed `
        --backbone uni `
        --patch-keep-frac 0.8 --attn-dispersion `
        --fold $fold --n-folds $NFolds --group-ts $GroupTs 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: M2_POOL_uni_SS_DISP fold=$fold" }
    Write-Host "=== M2_POOL_uni_SS_DISP fold=$fold/$NFolds Complete: $(Get-Date) ==="
}
Write-Host "=== ALL M2_POOL_uni_SS_DISP LOCAL 5-FOLD RUNS COMPLETE: $(Get-Date) ==="
