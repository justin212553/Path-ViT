$condaExe = Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"
(& $condaExe "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate PathViT-ray
Set-Location "D:\wonse\Documents\Job\urban_datalab\PATH-ViT"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
Remove-Item Env:\SSL_CERT_FILE -ErrorAction SilentlyContinue

# diagnose_wsi_reliance.py/diagnose_wsi_gradients.py 진단(findings_backlog.md 최상위 발견 2차) -
# risk_head가 z_rna 원본 벡터로 직결 우회하는 지름길이 WSI 브랜치의 gradient 기아 상태의
# 원인일 수 있다는 가설로, z_rna를 co-attention query로만 남기고 risk_head concat에서는 뺀
# --rna-gate-only(models/vit_pma.py rna_gate_only)를 PMA_EX_SS_AUX 기준으로 검증한다.

$LogDir = ".logs"
$Seeds = @(42, 84, 126)
$GroupTs = "0721pma_rnagate_ext"

$Total = $Seeds.Count
$Run = 0

foreach ($seed in $Seeds) {
    $Run++
    Write-Host "=== [$Run/$Total] PMA_EX_SS_AUX_RNAGATE seed=$seed Start: $(Get-Date) ==="
    $log = Join-Path $LogDir "train_tcga_seed${seed}_PMA_EX_SS_AUX_rnagate_ext.log"
    python -u .\train.py --dataset tcga --seed $seed --PMA --rna-genes literature_1500 `
        --patch-keep-frac 0.8 --rna-aux-weight 1.0 --rna-gate-only `
        --external --group-ts $GroupTs 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: PMA_EX_SS_AUX_RNAGATE seed=$seed" }
    Write-Host "=== [$Run/$Total] PMA_EX_SS_AUX_RNAGATE seed=$seed Complete: $(Get-Date) ==="
}
Write-Host "=== ALL PMA_EX_SS_AUX_RNAGATE RUNS COMPLETE: $(Get-Date) ==="
