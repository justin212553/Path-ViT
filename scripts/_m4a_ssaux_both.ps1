$condaExe = Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"
(& $condaExe "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate PathViT-ray
Set-Location "D:\wonse\Documents\Job\urban_datalab\PATH-ViT"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$LogDir = ".logs"
$Seeds = @(42, 84, 126)
$GroupTs = "0718m4assauxboth"

$Total = $Seeds.Count
$Run = 0

foreach ($seed in $Seeds) {
    $Run++
    Write-Host "=== [$Run/$Total] M4A_EX_SS_AUX both seed=$seed Start: $(Get-Date) ==="
    $log = Join-Path $LogDir "train_both_seed${seed}_M4A_EX_SS_AUX_both.log"
    python -u .\train.py --dataset both --seed $seed --M4A --rna-genes literature_1500 --patch-keep-frac 0.8 --rna-aux-weight 1.0 --group-ts $GroupTs | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: M4A_EX_SS_AUX both seed=$seed" }
    Write-Host "=== [$Run/$Total] M4A_EX_SS_AUX both seed=$seed Complete: $(Get-Date) ==="
}
Write-Host "=== ALL M4A_EX_SS_AUX BOTH RUNS COMPLETE: $(Get-Date) ==="
