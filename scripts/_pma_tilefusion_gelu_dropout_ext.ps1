$condaExe = Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"
(& $condaExe "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate PathViT-ray
Set-Location "D:\wonse\Documents\Job\urban_datalab\PATH-ViT"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# models/cnn_encoder.py::CNNEncoder.proj에 레퍼런스 tile_fusion과 동일하게 GELU+Dropout(0.4)
# 추가(기존엔 Linear+LayerNorm뿐) 후 PMA_EX_SS_AUX로 단독 검증. RNA 전처리 수정은 이미 메인
# 파이프라인에 반영된 상태(RNAFIX). external은 tcga train -> cptac test, 3시드.

$LogDir = ".logs"
$Seeds = @(42, 84, 126)
$GroupTs = "0721pma_tilefusion_gelu_dropout_ext"

$Total = $Seeds.Count
$Run = 0

foreach ($seed in $Seeds) {
    $Run++
    Write-Host "=== [$Run/$Total] PMA_EX_SS_AUX_TILEFUSION seed=$seed Start: $(Get-Date) ==="
    $log = Join-Path $LogDir "train_tcga_seed${seed}_PMA_EX_SS_AUX_TILEFUSION_ext.log"
    python -u .\train.py --dataset tcga --seed $seed --PMA --rna-genes literature_1500 `
        --patch-keep-frac 0.8 --rna-aux-weight 1.0 `
        --external --group-ts $GroupTs 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: PMA_EX_SS_AUX_TILEFUSION seed=$seed" }
    Write-Host "=== [$Run/$Total] PMA_EX_SS_AUX_TILEFUSION seed=$seed Complete: $(Get-Date) ==="
}
Write-Host "=== ALL PMA_EX_SS_AUX_TILEFUSION EXTERNAL RUNS COMPLETE: $(Get-Date) ==="
