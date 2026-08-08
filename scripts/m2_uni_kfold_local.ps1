$condaExe = Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"
(& $condaExe "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate PathViT-ray
Set-Location "D:\wonse\Documents\Job\urban_datalab\PATH-ViT"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
Remove-Item Env:\SSL_CERT_FILE -ErrorAction SilentlyContinue

# M2(WSI + Clinical age/sex late fusion, ABMIL)를 UNI 백본으로 5-fold external 검증.
# m1_uni_kfold_local.ps1과 동일한 이유(AUG는 UNI에서 external을 떨어뜨리는 게 확인됨)로 AUG 없이
# 돈다. precomputed features_uni.pt 사용(--image 불필요). M1/M2는 PMA 비교군이라 external c>0.5만
# 확인하면 된다.
#
# 2026-08-07(2차): --combine-mode cox_add 추가 — PMA(concat->cox_add)에서 clinical(margin/staging)
# 간섭 문제를 풀었던 방식이 M2(ABMIL, age/sex만) 구조에서도 재현되는지 궁금하다는 요청으로,
# models/vit_m2.py에 combine_mode="cox_add" 지원을 새로 이식(ViT_PMA와 동일 관례 — clinical을
# risk_head에 concat하지 않고 clinical_linear로 최종 스칼라에 zero-init 가산항으로 더함).
# model_prefix에 자동으로 _COX_ADD가 붙는다(train.py 공통 로직).
#
# 집계: python scripts/summarize_kfold.py --dataset tcga --seed 84 --n-folds 5 --model M2_uni_SS_DISP_COX_ADD

$LogDir = ".logs"
$Seed = 84
$NFolds = 5
$GroupTs = "0807m2_uni_coxadd_kfold5_local_seed84"

for ($fold = 0; $fold -lt $NFolds; $fold++) {
    Write-Host "=== M2_uni_SS_DISP_COX_ADD fold=$fold/$NFolds Start: $(Get-Date) ==="
    $log = Join-Path $LogDir "train_tcga_seed${Seed}_M2_uni_SS_DISP_COX_ADD_kfold5_fold${fold}.log"
    python -u .\train.py --M2 --dataset tcga --external --seed $Seed `
        --backbone uni --combine-mode cox_add `
        --patch-keep-frac 0.8 --attn-dispersion `
        --fold $fold --n-folds $NFolds --group-ts $GroupTs 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: M2_uni_SS_DISP_COX_ADD fold=$fold" }
    Write-Host "=== M2_uni_SS_DISP_COX_ADD fold=$fold/$NFolds Complete: $(Get-Date) ==="
}
Write-Host "=== ALL M2_uni_SS_DISP_COX_ADD LOCAL 5-FOLD RUNS COMPLETE: $(Get-Date) ==="
