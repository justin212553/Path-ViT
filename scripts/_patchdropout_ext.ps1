$condaExe = Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"
(& $condaExe "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate PathViT-ray
Set-Location "D:\wonse\Documents\Job\urban_datalab\PATH-ViT"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$LogDir = ".logs"
$Seeds = @(42, 84, 126)
$Datasets = @("tcga", "cptac")
$GroupTs = "0717ss"
$KeepFrac = "0.8"

$Models = @("--M4A", "--PMA")
$Tags   = @("M4A", "PMA")

$Total = $Models.Count * $Datasets.Count * $Seeds.Count
$Run = 0

for ($i = 0; $i -lt $Models.Count; $i++) {
    $flag = $Models[$i]; $tag = $Tags[$i]
    foreach ($ds in $Datasets) {
        foreach ($seed in $Seeds) {
            $Run++
            Write-Host "=== [$Run/$Total] ${tag}_SS ds=$ds seed=$seed Start: $(Get-Date) ==="
            $log = Join-Path $LogDir "train_${ds}_seed${seed}_${tag}_EX_SS_ext.log"
            python -u .\train.py --dataset $ds --seed $seed $flag --rna-genes literature_1500 --patch-keep-frac $KeepFrac --external --group-ts $GroupTs | Tee-Object -FilePath $log
            if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: ${tag}_SS ds=$ds seed=$seed" }
            Write-Host "=== [$Run/$Total] ${tag}_SS ds=$ds seed=$seed Complete: $(Get-Date) ==="
        }
    }
}
Write-Host "=== ALL PATCHDROPOUT EXTERNAL RUNS COMPLETE: $(Get-Date) ==="
