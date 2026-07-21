$condaExe = Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"
(& $condaExe "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate PathViT-ray
Set-Location "D:\wonse\Documents\Job\urban_datalab\PATH-ViT"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# 진짜 마지막 ablation: PMA_EX_SS_AUX(지금까지 external 최고 기록, 0.619) 기준에서 Nystromformer
# FFN 서브레이어까지 제거해본다(8번 항목 M4A_FF와 같은 논리, PMA 기준으로는 처음 시도).
# 기존(재타일링 아닌) 패치 사용 - 재타일링은 이미 negative result로 종료됨(4번 항목).

$LogDir = ".logs"
$Seeds = @(42, 84, 126)
$Datasets = @("tcga", "cptac")
$GroupTs = "0719pma_ff_ssaux"

$Total = $Datasets.Count * $Seeds.Count
$Run = 0

foreach ($ds in $Datasets) {
    foreach ($seed in $Seeds) {
        $Run++
        Write-Host "=== [$Run/$Total] PMA_FF_EX_SS_AUX ds=$ds seed=$seed Start: $(Get-Date) ==="
        $log = Join-Path $LogDir "train_${ds}_seed${seed}_PMA_FF_EX_SS_AUX_ext.log"
        python -u .\train.py --dataset $ds --seed $seed --PMA_FF --rna-genes literature_1500 `
            --patch-keep-frac 0.8 --rna-aux-weight 1.0 --external --group-ts $GroupTs 2>&1 | Tee-Object -FilePath $log
        if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: PMA_FF_EX_SS_AUX ds=$ds seed=$seed" }
        Write-Host "=== [$Run/$Total] PMA_FF_EX_SS_AUX ds=$ds seed=$seed Complete: $(Get-Date) ==="
    }
}
Write-Host "=== ALL PMA_FF_EX_SS_AUX EXTERNAL RUNS COMPLETE: $(Get-Date) ==="
