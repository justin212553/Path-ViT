$condaExe = Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"
(& $condaExe "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate PathViT-ray
Set-Location "D:\wonse\Documents\Job\urban_datalab\PATH-ViT"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# config.py 2026-07-19 변경: embed_dim 64->256, num_heads 2->4, num_transformer_layers 1->2
# ("WSI 브랜치가 과적합이 아니라 과압축이라 신호를 못 살렸을 수도 있다" 가설 검증).
# 앞으로 external은 tcga train / cptac test 방향으로 고정(사용자 지시, 2026-07-19).

$LogDir = ".logs"
$Seeds = @(42, 84, 126)
$GroupTs = "0719pma_bigcap"

$Total = $Seeds.Count
$Run = 0

foreach ($seed in $Seeds) {
    $Run++
    Write-Host "=== [$Run/$Total] PMA_EX_SS_AUX(bigcap, tcga->cptac) seed=$seed Start: $(Get-Date) ==="
    $log = Join-Path $LogDir "train_tcga_seed${seed}_PMA_EX_SS_AUX_bigcap_ext.log"
    python -u .\train.py --dataset tcga --seed $seed --PMA --rna-genes literature_1500 `
        --patch-keep-frac 0.8 --rna-aux-weight 1.0 --external --group-ts $GroupTs 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: PMA_EX_SS_AUX(bigcap) seed=$seed" }
    Write-Host "=== [$Run/$Total] PMA_EX_SS_AUX(bigcap, tcga->cptac) seed=$seed Complete: $(Get-Date) ==="
}
Write-Host "=== ALL PMA_EX_SS_AUX BIGCAP EXTERNAL RUNS COMPLETE: $(Get-Date) ==="
