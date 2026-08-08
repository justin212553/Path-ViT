$condaExe = Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"
(& $condaExe "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate PathViT-ray
Set-Location "D:\wonse\Documents\Job\urban_datalab\PATH-ViT"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
Remove-Item Env:\SSL_CERT_FILE -ErrorAction SilentlyContinue

# M1_POOL(WSI 단독, MultiComponentPooling(mean/std/attn/top-k) + self-attention)을 UNI 백본으로
# 5-fold external 검증. M1(ABMIL)/M2(ABMIL)가 UNI에서 external 0.44~0.49대(사실상 랜덤 이하)로
# 나온 뒤, "ABMIL 자체가 UNI의 정보를 못 우려내는 게 아닌가"라는 가설로 M3/PMA와 동일한
# MultiComponentPooling을 이식한다. 다만 M1엔 RNA/clinical처럼 "어디를 봐야 하는지" 알려줄
# 외부 모달리티가 없으므로, 학습되는 고정 query로 co-attention하는 대신 4개 관점이 서로
# self-attention하도록 설계했다(models/vit_m1_pool.py 2026-08-07 재작성).
# AUG는 UNI에서 external을 떨어뜨리는 게 확인됐으므로 빼고 돈다. precomputed features_uni.pt
# 사용(--image 불필요). 최소 기준은 external c>0.5.
#
# 집계: python scripts/summarize_kfold.py --dataset tcga --seed 84 --n-folds 5 --model M1_POOL_uni_SS_DISP

$LogDir = ".logs"
$Seed = 84
$NFolds = 5
$GroupTs = "0807m1pool_uni_kfold5_local_seed84"

for ($fold = 0; $fold -lt $NFolds; $fold++) {
    Write-Host "=== M1_POOL_uni_SS_DISP fold=$fold/$NFolds Start: $(Get-Date) ==="
    $log = Join-Path $LogDir "train_tcga_seed${Seed}_M1_POOL_uni_SS_DISP_kfold5_fold${fold}.log"
    python -u .\train.py --M1_POOL --dataset tcga --external --seed $Seed `
        --backbone uni `
        --patch-keep-frac 0.8 --attn-dispersion `
        --fold $fold --n-folds $NFolds --group-ts $GroupTs 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: M1_POOL_uni_SS_DISP fold=$fold" }
    Write-Host "=== M1_POOL_uni_SS_DISP fold=$fold/$NFolds Complete: $(Get-Date) ==="
}
Write-Host "=== ALL M1_POOL_uni_SS_DISP LOCAL 5-FOLD RUNS COMPLETE: $(Get-Date) ==="
