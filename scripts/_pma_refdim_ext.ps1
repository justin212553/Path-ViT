$condaExe = Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"
(& $condaExe "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate PathViT-ray
Set-Location "D:\wonse\Documents\Job\urban_datalab\PATH-ViT"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# models/vit_pma.py: RNA 인코더를 레퍼런스 사양(RNAEncoderExtend, 256dim)으로, Clinical을
# 16dim으로 교체(기존엔 셋 다 cfg.embed_dim으로 억지로 맞췄음). tile-fusion(GELU+Dropout,
# cnn_encoder.py::proj)은 이미 적용된 상태 위에 얹는다. PMA_EX_SS_AUX 기준, RNA 전처리도
# 이미 수정된 상태. external은 tcga train -> cptac test, 3시드.

$LogDir = ".logs"
$Seeds = @(42, 84, 126)
$GroupTs = "0721pma_refdim_ext"

$Total = $Seeds.Count
$Run = 0

foreach ($seed in $Seeds) {
    $Run++
    Write-Host "=== [$Run/$Total] PMA_EX_SS_AUX_REFDIM seed=$seed Start: $(Get-Date) ==="
    $log = Join-Path $LogDir "train_tcga_seed${seed}_PMA_EX_SS_AUX_REFDIM_ext.log"
    python -u .\train.py --dataset tcga --seed $seed --PMA --rna-genes literature_1500 `
        --patch-keep-frac 0.8 --rna-aux-weight 1.0 `
        --external --group-ts $GroupTs 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: PMA_EX_SS_AUX_REFDIM seed=$seed" }
    Write-Host "=== [$Run/$Total] PMA_EX_SS_AUX_REFDIM seed=$seed Complete: $(Get-Date) ==="
}
Write-Host "=== ALL PMA_EX_SS_AUX_REFDIM EXTERNAL RUNS COMPLETE: $(Get-Date) ==="
