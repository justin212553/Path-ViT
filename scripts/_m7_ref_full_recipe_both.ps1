$condaExe = Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"
(& $condaExe "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate PathViT-ray
Set-Location "D:\wonse\Documents\Job\urban_datalab\PATH-ViT"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# 레퍼런스(Leeyoungsup/pancreatic_cancer_pathology) M7을 최대한 그대로 재현:
# RNA=256(LayerNorm+Dropout 정규화), Clinical=16, lr=5e-5, weight_decay=1e-3,
# epochs=100 + early stopping patience=20, --dataset both(레퍼런스와 동일한 pooled 프로토콜 -
# GitHub M4_Train.ipynb의 train_test_split(stratify=dataset+event)로 확인됨).

$LogDir = ".logs"
$Seeds = @(42, 84, 126)
$GroupTs = "0719m7fullrecipe_both"
$Lr = "5e-5"
$Wd = "1e-3"
$Epochs = 100
$Patience = 20

$Total = $Seeds.Count
$Run = 0

foreach ($seed in $Seeds) {
    $Run++
    Write-Host "=== [$Run/$Total] M7_EX_fullrecipe both seed=$seed Start: $(Get-Date) ==="
    $log = Join-Path $LogDir "train_light_both_seed${seed}_M7_EX_fullrecipe_both.log"
    python -u .\train_light.py --dataset both --seed $seed --M7 --rna-genes literature_1500 `
        --lr $Lr --weight-decay $Wd --epochs $Epochs --patience $Patience --group-ts $GroupTs 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: M7_EX_fullrecipe both seed=$seed" }
    Write-Host "=== [$Run/$Total] M7_EX_fullrecipe both seed=$seed Complete: $(Get-Date) ==="
}
Write-Host "=== ALL M7_EX FULLRECIPE BOTH RUNS COMPLETE: $(Get-Date) ==="
