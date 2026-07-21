$condaExe = Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"
(& $condaExe "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate PathViT-ray
Set-Location "D:\wonse\Documents\Job\urban_datalab\PATH-ViT"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# M7_EX(literature_1500) risk_head를 레퍼런스(tabular_survival.py::ClinicalRNASeqSurvivalModel)
# 사양(LayerNorm->Dropout0.4->Linear(272->128)->GELU->Dropout0.4->Linear(128->1))으로 교체한
# 버전을 기존 M7_EX 기본 레시피(lr/wd/epochs 오버라이드 없음, --match-reference-cohort 없음)
# 그대로 --external(2코호트x3시드)로 재검증. 기존 M7_EX 기준(external C 평균 0.634)과 직접 비교.

$LogDir = ".logs"
$Datasets = @("tcga", "cptac")
$Seeds = @(42, 84, 126)
$GroupTs = "0721m7riskhead_ext"

$Total = $Datasets.Count * $Seeds.Count
$Run = 0

foreach ($dataset in $Datasets) {
    foreach ($seed in $Seeds) {
        $Run++
        Write-Host "=== [$Run/$Total] M7_EX_riskhead $dataset seed=$seed Start: $(Get-Date) ==="
        $log = Join-Path $LogDir "train_light_${dataset}_seed${seed}_M7_EX_riskhead_ext.log"
        python -u .\train_light.py --dataset $dataset --seed $seed --M7 --rna-genes literature_1500 `
            --external --group-ts $GroupTs 2>&1 | Tee-Object -FilePath $log
        if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: M7_EX_riskhead $dataset seed=$seed" }
        Write-Host "=== [$Run/$Total] M7_EX_riskhead $dataset seed=$seed Complete: $(Get-Date) ==="
    }
}
Write-Host "=== ALL M7_EX RISKHEAD EXTERNAL RUNS COMPLETE: $(Get-Date) ==="
