$condaExe = Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"
(& $condaExe "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate PathViT-ray
Set-Location "D:\wonse\Documents\Job\urban_datalab\PATH-ViT"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
Remove-Item Env:\SSL_CERT_FILE -ErrorAction SilentlyContinue

# --full-attention(나이스트롬 -> 표준 O(N^2) attention) seed42 결과가 internal은 개선(+0.026),
# external은 악화(-0.028)로 갈렸다(findings_backlog.md) - 1시드만으로는 판단 이르므로 84/126도
# 확인한다. PMA_EX_SS_AUX 기준, external.

$LogDir = ".logs"
$Seeds = @(84, 126)
$GroupTs = "0722pma_fullattn_ext"

$Total = $Seeds.Count
$Run = 0

foreach ($seed in $Seeds) {
    $Run++
    Write-Host "=== [$Run/$Total] PMA_EX_SS_AUX_FULLATTN seed=$seed Start: $(Get-Date) ==="
    $log = Join-Path $LogDir "train_tcga_seed${seed}_PMA_EX_SS_AUX_fullattn_ext.log"
    python -u .\train.py --dataset tcga --seed $seed --PMA --rna-genes literature_1500 `
        --patch-keep-frac 0.8 --rna-aux-weight 1.0 --full-attention `
        --external --group-ts $GroupTs 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: PMA_EX_SS_AUX_FULLATTN seed=$seed" }
    Write-Host "=== [$Run/$Total] PMA_EX_SS_AUX_FULLATTN seed=$seed Complete: $(Get-Date) ==="
}
Write-Host "=== ALL FULLATTN SEEDS84-126 RUNS COMPLETE: $(Get-Date) ==="
