$condaExe = Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"
(& $condaExe "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate PathViT-ray
Set-Location "D:\wonse\Documents\Job\urban_datalab\PATH-ViT"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
Remove-Item Env:\SSL_CERT_FILE -ErrorAction SilentlyContinue

# utils/extract_features_augmented.py로 TCGA 전체 코호트(377슬라이드, seed=42 augmentation)를
# 미리 뽑아둔 features_aug.pt를 --tile-augment로 읽는다 — train split에서만 적용되고
# val/test/external은 항상 원본 features.pt(증강 없음). PMA_EX_SS_AUX 기준 external
# (tcga train -> cptac test), 3시드.

$LogDir = ".logs"
$Seeds = @(42, 84, 126)
$GroupTs = "0721pma_tileaugment_local_ext"

$Total = $Seeds.Count
$Run = 0

foreach ($seed in $Seeds) {
    $Run++
    Write-Host "=== [$Run/$Total] PMA_EX_SS_AUX_AUG seed=$seed Start: $(Get-Date) ==="
    $log = Join-Path $LogDir "train_tcga_seed${seed}_PMA_EX_SS_AUX_AUG_ext.log"
    python -u .\train.py --dataset tcga --seed $seed --PMA --rna-genes literature_1500 `
        --patch-keep-frac 0.8 --rna-aux-weight 1.0 --tile-augment `
        --external --group-ts $GroupTs 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: PMA_EX_SS_AUX_AUG seed=$seed" }
    Write-Host "=== [$Run/$Total] PMA_EX_SS_AUX_AUG seed=$seed Complete: $(Get-Date) ==="
}
Write-Host "=== ALL PMA_EX_SS_AUX_AUG LOCAL EXTERNAL RUNS COMPLETE: $(Get-Date) ==="
