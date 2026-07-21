$condaExe = Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"
(& $condaExe "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate PathViT-ray
Set-Location "D:\wonse\Documents\Job\urban_datalab\PATH-ViT"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# findings_backlog.md 4번 항목 핵심 검증: 재타일링(512px@0.5MPP, data/patches_{tcga,cptac}_512)이
# 실제로 WSI 모델의 external 성능을 끌어올리는지 - 지금까지 external 최고 기록인
# PMA_EX_SS_AUX(기존 타일링 기준 external C=0.619, M7_EX 0.634에는 못 미침) 하나만 가볍게
# 단일 시드로 먼저 확인해본다.

$LogDir = ".logs"
$Seed = 42
$Datasets = @("tcga", "cptac")
$GroupTs = "0719pmassaux_retiled"

$Total = $Datasets.Count
$Run = 0

foreach ($ds in $Datasets) {
    $Run++
    Write-Host "=== [$Run/$Total] PMA_EX_SS_AUX(retiled) ds=$ds seed=$Seed Start: $(Get-Date) ==="
    $log = Join-Path $LogDir "train_${ds}_seed${Seed}_PMA_EX_SS_AUX_retiled_ext.log"
    python -u .\train.py --dataset $ds --seed $Seed --PMA --rna-genes literature_1500 `
        --patch-keep-frac 0.8 --rna-aux-weight 1.0 `
        --patches-root-tcga data/patches_tcga_512 --patches-root-cptac data/patches_cptac_512 `
        --external --group-ts $GroupTs | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: PMA_EX_SS_AUX(retiled) ds=$ds seed=$Seed" }
    Write-Host "=== [$Run/$Total] PMA_EX_SS_AUX(retiled) ds=$ds seed=$Seed Complete: $(Get-Date) ==="
}
Write-Host "=== ALL PMA_EX_SS_AUX RETILED EXTERNAL RUNS COMPLETE: $(Get-Date) ==="
