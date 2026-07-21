$condaExe = Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"
(& $condaExe "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate PathViT-ray
Set-Location "D:\wonse\Documents\Job\urban_datalab\PATH-ViT"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# M7(ClinicalRNAOnly)을 레퍼런스(Leeyoungsup/pancreatic_cancer_pathology) 사양에 맞춘 버전:
# RNA 인코더 = RNAEncoderExtend(G->256->256, LayerNorm+Dropout 입출력 정규화, GitHub 원문 이식),
# Clinical = 16차원 그대로(레퍼런스와 동일 비대칭), lr=5e-5/weight_decay=1e-3(레퍼런스 M7 레시피).
# 두 프로토콜 다 확인: --dataset both(레퍼런스 0.701과 동일 프로토콜, apples-to-apples)와
# --external(tcga train/cptac test 고정, 우리 프로젝트의 진짜 기준).

$LogDir = ".logs"
$Seeds = @(42, 84, 126)
$GroupTs = "0719m7refrecipe"
$Lr = "5e-5"
$Wd = "1e-3"

$Total = $Seeds.Count * 2
$Run = 0

foreach ($seed in $Seeds) {
    $Run++
    Write-Host "=== [$Run/$Total] M7_EX_refrecipe both seed=$seed Start: $(Get-Date) ==="
    $log = Join-Path $LogDir "train_light_both_seed${seed}_M7_EX_refrecipe_both.log"
    python -u .\train_light.py --dataset both --seed $seed --M7 --rna-genes literature_1500 --lr $Lr --weight-decay $Wd --group-ts $GroupTs 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: M7_EX_refrecipe both seed=$seed" }
    Write-Host "=== [$Run/$Total] M7_EX_refrecipe both seed=$seed Complete: $(Get-Date) ==="
}

foreach ($seed in $Seeds) {
    $Run++
    Write-Host "=== [$Run/$Total] M7_EX_refrecipe tcga->cptac seed=$seed Start: $(Get-Date) ==="
    $log = Join-Path $LogDir "train_light_tcga_seed${seed}_M7_EX_refrecipe_ext.log"
    python -u .\train_light.py --dataset tcga --seed $seed --M7 --rna-genes literature_1500 --lr $Lr --weight-decay $Wd --external --group-ts $GroupTs 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: M7_EX_refrecipe ext seed=$seed" }
    Write-Host "=== [$Run/$Total] M7_EX_refrecipe tcga->cptac seed=$seed Complete: $(Get-Date) ==="
}
Write-Host "=== ALL M7_EX REFRECIPE RUNS COMPLETE: $(Get-Date) ==="
