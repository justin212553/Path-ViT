$condaExe = Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"
(& $condaExe "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate PathViT-ray
Set-Location "D:\wonse\Documents\Job\urban_datalab\PATH-ViT"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# findings_backlog.md 13번 항목 후속: 레퍼런스 M4(m4_pathology_rnaseq_clinical_mil.py::classifier)의
# risk head는 은닉층 없이 LayerNorm->Dropout(0.4)->Linear(->1)뿐이다(레퍼런스 M7의 은닉층+GELU
# 버전과 다름 - 그건 이미 negative result로 확인됨). models/vit_pma.py risk_head에 Dropout(0.4)만
# 추가(파라미터 증가 없음, 순수 규제)한 버전을 PMA_EX_SS_AUX 기준으로 재검증.
# external은 이 세션의 표준 관례대로 tcga train -> cptac test 단일 방향만 돈다.

$LogDir = ".logs"
$Datasets = @("tcga")
$Seeds = @(42, 84, 126)
$GroupTs = "0721pma_riskheaddrop_ext"

$Total = $Datasets.Count * $Seeds.Count
$Run = 0

foreach ($dataset in $Datasets) {
    foreach ($seed in $Seeds) {
        $Run++
        Write-Host "=== [$Run/$Total] PMA_EX_SS_AUX_RiskHeadDrop $dataset seed=$seed Start: $(Get-Date) ==="
        $log = Join-Path $LogDir "train_${dataset}_seed${seed}_PMA_EX_SS_AUX_riskheaddrop_ext.log"
        python -u .\train.py --dataset $dataset --seed $seed --PMA --rna-genes literature_1500 `
            --patch-keep-frac 0.8 --rna-aux-weight 1.0 `
            --external --group-ts $GroupTs 2>&1 | Tee-Object -FilePath $log
        if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: PMA_EX_SS_AUX_RiskHeadDrop $dataset seed=$seed" }
        Write-Host "=== [$Run/$Total] PMA_EX_SS_AUX_RiskHeadDrop $dataset seed=$seed Complete: $(Get-Date) ==="
    }
}
Write-Host "=== ALL PMA_EX_SS_AUX_RISKHEADDROP EXTERNAL RUNS COMPLETE: $(Get-Date) ==="
