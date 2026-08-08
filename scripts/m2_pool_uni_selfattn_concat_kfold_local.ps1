$condaExe = Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"
(& $condaExe "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate PathViT-ray
Set-Location "D:\wonse\Documents\Job\urban_datalab\PATH-ViT"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
Remove-Item Env:\SSL_CERT_FILE -ErrorAction SilentlyContinue

# M2_POOL(pooling_mode=selfattn, combine_mode=concat) — 4개 pooling 관점은 M1_POOL과 동일한
# self-attention(clinical 개입 없음)으로 합치고, clinical(age/sex)은 ClinicalEncoder의 완전한
# MLP 임베딩으로 concat한다. co-attention query(external 0.497)와 cox_add(external 0.500) 둘 다
# M1_POOL 단독(0.556)보다 낮았던 뒤 시도하는 세 번째 조합 — concat은 clinical이 "싼 지름길"이
# 아니라 pooling만큼 비용을 들여 학습해야 하는 임베딩이라 shortcut-learning 문제를 피할 수
# 있을 것이라는 가설. AUG는 UNI에서 계속 손해라 빼고, precomputed features_uni.pt 사용.
#
# 집계: python scripts/summarize_kfold.py --dataset tcga --seed 84 --n-folds 5 --model M2_POOL_uni_SS_DISP_SELFATTN

$LogDir = ".logs"
$Seed = 84
$NFolds = 5
$GroupTs = "0807m2pool_uni_selfattn_concat_kfold5_local_seed84"

for ($fold = 0; $fold -lt $NFolds; $fold++) {
    Write-Host "=== M2_POOL_uni_SS_DISP_SELFATTN fold=$fold/$NFolds Start: $(Get-Date) ==="
    $log = Join-Path $LogDir "train_tcga_seed${Seed}_M2_POOL_uni_SS_DISP_SELFATTN_kfold5_fold${fold}.log"
    python -u .\train.py --M2_POOL --dataset tcga --external --seed $Seed `
        --backbone uni --pooling-mode selfattn `
        --patch-keep-frac 0.8 --attn-dispersion `
        --fold $fold --n-folds $NFolds --group-ts $GroupTs 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: M2_POOL_uni_SS_DISP_SELFATTN fold=$fold" }
    Write-Host "=== M2_POOL_uni_SS_DISP_SELFATTN fold=$fold/$NFolds Complete: $(Get-Date) ==="
}
Write-Host "=== ALL M2_POOL_uni_SS_DISP_SELFATTN LOCAL 5-FOLD RUNS COMPLETE: $(Get-Date) ==="
