$condaExe = Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"
(& $condaExe "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate PathViT-ray
Set-Location "D:\wonse\Documents\Job\urban_datalab\PATH-ViT"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$LogDir = ".logs"
$Seeds = @(42, 84, 126)
$Datasets = @("tcga", "cptac")
$GroupTs = "0717resume"

$Task1Models = @("--M4", "--M4A", "--PM4")
$Task1Tags   = @("M4", "M4A", "PM4")
$Task2Models = @("--M6", "--M7")
$Task2Tags   = @("M6", "M7")

# 이미 완료된 run 1개 (M4 tcga seed42) 는 건너뛴다.
$AlreadyDone = @("M4|tcga|42")

$Total = ($Task1Models.Count + $Task2Models.Count) * $Datasets.Count * $Seeds.Count
$Run = 0

Write-Host "=== EXTERNAL VALIDATION RESUME: M4_EX/M4A_EX/PM4_EX (train.py) ==="
for ($i = 0; $i -lt $Task1Models.Count; $i++) {
    $flag = $Task1Models[$i]; $tag = $Task1Tags[$i]
    foreach ($ds in $Datasets) {
        foreach ($seed in $Seeds) {
            $Run++
            $key = "$tag|$ds|$seed"
            if ($AlreadyDone -contains $key) {
                Write-Host "=== [$Run/$Total] $tag ds=$ds seed=$seed SKIP (already done) ==="
                continue
            }
            Write-Host "=== [$Run/$Total] $tag ds=$ds seed=$seed Start: $(Get-Date) ==="
            $log = Join-Path $LogDir "train_${ds}_seed${seed}_${tag}_EX_ext.log"
            python -u .\train.py --dataset $ds --seed $seed $flag --rna-genes literature_1500 --external --group-ts $GroupTs | Tee-Object -FilePath $log
            if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $tag ds=$ds seed=$seed" }
            Write-Host "=== [$Run/$Total] $tag ds=$ds seed=$seed Complete: $(Get-Date) ==="
        }
    }
}

Write-Host "=== EXTERNAL VALIDATION RESUME: M6_EX/M7_EX (train_light.py) ==="
for ($i = 0; $i -lt $Task2Models.Count; $i++) {
    $flag = $Task2Models[$i]; $tag = $Task2Tags[$i]
    foreach ($ds in $Datasets) {
        foreach ($seed in $Seeds) {
            $Run++
            Write-Host "=== [$Run/$Total] $tag ds=$ds seed=$seed Start: $(Get-Date) ==="
            $log = Join-Path $LogDir "train_light_${ds}_seed${seed}_${tag}_EX_ext.log"
            python -u .\train_light.py --dataset $ds --seed $seed $flag --rna-genes literature_1500 --external --group-ts $GroupTs | Tee-Object -FilePath $log
            if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $tag ds=$ds seed=$seed" }
            Write-Host "=== [$Run/$Total] $tag ds=$ds seed=$seed Complete: $(Get-Date) ==="
        }
    }
}
Write-Host "=== ALL EXTERNAL VALIDATION COMPLETE: $(Get-Date) ==="
