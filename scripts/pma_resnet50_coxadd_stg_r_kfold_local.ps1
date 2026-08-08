$condaExe = Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"
(& $condaExe "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate PathViT-ray
Set-Location "D:\wonse\Documents\Job\urban_datalab\PATH-ViT"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
Remove-Item Env:\SSL_CERT_FILE -ErrorAction SilentlyContinue

# PMA(ResNet50, INT1500, margin+staging, cox_add) no-aug 5-fold — 지금까지의 최고 기록
# (PMA_uni_INT1500_SS_AUX_STG_R_DISP_COX_ADD, external 0.644)이 UNI 때문인지 staging 때문인지
# 분리하는 검증. 같은 레시피에서 margin만 있고 staging 없는 버전은 이미 backbone 통제 비교로
# ResNet50=0.616/UNI=0.621(거의 노이즈 수준 차이)임을 확인했다(scripts/compare_pma_backbone_ext.py)
# — staging을 마저 추가했을 때 ResNet50도 UNI의 0.644에 근접하는지가 이번 질문의 핵심.
# precomputed features.pt(ResNet50) 이미 존재해 --image 불필요, 빠르게 돈다.
#
# 집계: python scripts/summarize_kfold.py --dataset tcga --seed 84 --n-folds 5 --model PMA_INT1500_SS_AUX_STG_R_DISP_COX_ADD

$LogDir = ".logs"
$Seed = 84
$NFolds = 5
$GroupTs = "0808pma_resnet50_coxadd_stg_r_kfold5_local_seed84"

for ($fold = 0; $fold -lt $NFolds; $fold++) {
    Write-Host "=== PMA_INT1500_SS_AUX_STG_R_DISP_COX_ADD(resnet50) fold=$fold/$NFolds Start: $(Get-Date) ==="
    $log = Join-Path $LogDir "train_tcga_seed${Seed}_PMA_INT1500_SS_AUX_STG_R_DISP_COX_ADD_kfold5_fold${fold}.log"
    python -u .\train.py --PMA --rna-genes literature_1500_intersection --dataset tcga --external --seed $Seed `
        --clinical-margin --clinical-staging --combine-mode cox_add `
        --patch-keep-frac 0.8 --attn-dispersion --rna-aux-weight 1.0 `
        --fold $fold --n-folds $NFolds --group-ts $GroupTs 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: PMA_INT1500_SS_AUX_STG_R_DISP_COX_ADD(resnet50) fold=$fold" }
    Write-Host "=== PMA_INT1500_SS_AUX_STG_R_DISP_COX_ADD(resnet50) fold=$fold/$NFolds Complete: $(Get-Date) ==="
}
Write-Host "=== ALL PMA_INT1500_SS_AUX_STG_R_DISP_COX_ADD(resnet50) LOCAL 5-FOLD RUNS COMPLETE: $(Get-Date) ==="
