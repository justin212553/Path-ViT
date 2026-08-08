$condaExe = Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"
(& $condaExe "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate PathViT-ray
Set-Location "D:\wonse\Documents\Job\urban_datalab\PATH-ViT"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
Remove-Item Env:\SSL_CERT_FILE -ErrorAction SilentlyContinue

# M2_POOL(clinical co-attention query, models/vit_m2_pool.py) UNI 5-fold 결과가 external
# 0.49~0.51(사실상 랜덤, M1_POOL의 self-attention 단독 0.556보다도 낮음)로 나온 뒤 —
# age/sex처럼 약한 신호를 co-attention query로 쓰는 게 오히려 pooling을 망치는 것 아니냐는
# 가설로, pooling은 M1_POOL과 동일한 self-attention(clinical 개입 없음)으로 하고 clinical은
# PMA/M2와 동일한 관례로 risk_head 스칼라에 cox_add(zero-init 가산항)로만 더하는 조합을 검증한다.
# AUG는 UNI에서 손해라 계속 빼고, precomputed features_uni.pt 사용(--image 불필요).
#
# 집계: python scripts/summarize_kfold.py --dataset tcga --seed 84 --n-folds 5 --model M2_POOL_uni_SS_DISP_SELFATTN_COX_ADD

$LogDir = ".logs"
$Seed = 84
$NFolds = 5
$GroupTs = "0807m2pool_uni_selfattn_coxadd_kfold5_local_seed84"

for ($fold = 0; $fold -lt $NFolds; $fold++) {
    Write-Host "=== M2_POOL_uni_SS_DISP_SELFATTN_COX_ADD fold=$fold/$NFolds Start: $(Get-Date) ==="
    $log = Join-Path $LogDir "train_tcga_seed${Seed}_M2_POOL_uni_SS_DISP_SELFATTN_COX_ADD_kfold5_fold${fold}.log"
    python -u .\train.py --M2_POOL --dataset tcga --external --seed $Seed `
        --backbone uni --pooling-mode selfattn --combine-mode cox_add `
        --patch-keep-frac 0.8 --attn-dispersion `
        --fold $fold --n-folds $NFolds --group-ts $GroupTs 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: M2_POOL_uni_SS_DISP_SELFATTN_COX_ADD fold=$fold" }
    Write-Host "=== M2_POOL_uni_SS_DISP_SELFATTN_COX_ADD fold=$fold/$NFolds Complete: $(Get-Date) ==="
}
Write-Host "=== ALL M2_POOL_uni_SS_DISP_SELFATTN_COX_ADD LOCAL 5-FOLD RUNS COMPLETE: $(Get-Date) ==="
