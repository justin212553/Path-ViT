$condaExe = Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"
(& $condaExe "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate PathViT-ray
Set-Location "D:\wonse\Documents\Job\urban_datalab\PATH-ViT"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# findings_backlog.md 13번 항목 후속: RNA 인코더의 "레이어 구조"(레퍼런스 LayerNorm+Dropout
# 입출력 정규화)는 유지하되 "폭"만 256->64로 되돌려서, 13번의 악화가 구조 때문인지 폭 때문인지
# 분리한다. Clinical=16, lr=5e-5/wd=1e-3(레퍼런스 레시피)는 13번과 동일하게 유지.

$LogDir = ".logs"
$Seeds = @(42, 84, 126)
$GroupTs = "0719m7reflayers_dim64"
$Lr = "5e-5"
$Wd = "1e-3"

$Total = $Seeds.Count * 2
$Run = 0

foreach ($seed in $Seeds) {
    $Run++
    Write-Host "=== [$Run/$Total] M7_EX_reflayers_dim64 both seed=$seed Start: $(Get-Date) ==="
    $log = Join-Path $LogDir "train_light_both_seed${seed}_M7_EX_reflayers_dim64_both.log"
    python -u .\train_light.py --dataset both --seed $seed --M7 --rna-genes literature_1500 --lr $Lr --weight-decay $Wd --group-ts $GroupTs 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: M7_EX_reflayers_dim64 both seed=$seed" }
    Write-Host "=== [$Run/$Total] M7_EX_reflayers_dim64 both seed=$seed Complete: $(Get-Date) ==="
}

foreach ($seed in $Seeds) {
    $Run++
    Write-Host "=== [$Run/$Total] M7_EX_reflayers_dim64 tcga->cptac seed=$seed Start: $(Get-Date) ==="
    $log = Join-Path $LogDir "train_light_tcga_seed${seed}_M7_EX_reflayers_dim64_ext.log"
    python -u .\train_light.py --dataset tcga --seed $seed --M7 --rna-genes literature_1500 --lr $Lr --weight-decay $Wd --external --group-ts $GroupTs 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: M7_EX_reflayers_dim64 ext seed=$seed" }
    Write-Host "=== [$Run/$Total] M7_EX_reflayers_dim64 tcga->cptac seed=$seed Complete: $(Get-Date) ==="
}
Write-Host "=== ALL M7_EX REFLAYERS DIM64 RUNS COMPLETE: $(Get-Date) ==="
