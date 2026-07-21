$condaExe = Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"
(& $condaExe "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate PathViT-ray
Set-Location "D:\wonse\Documents\Job\urban_datalab\PATH-ViT"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# models/clinical_rna_only.py risk_head에 레퍼런스 M4 사양대로 은닉층 없이 Dropout(0.4)만
# 추가 - RNA 전처리 수정 이후 M7_EX(기본 레시피)로 재검증. (이전 시도는 $GroupTs1 변수명
# 관련 PowerShell 파싱 버그로 즉시 실패했음 - $GroupTs로 통일해 재시도.)

$LogDir = ".logs"
$Seeds = @(42, 84, 126)
$GroupTs = "0721m7_riskhead_dropout_only_v2_ext"

foreach ($seed in $Seeds) {
    Write-Host "=== M7_EX_RISKHEADDROPONLY seed=$seed Start: $(Get-Date) ==="
    $log = Join-Path $LogDir "train_light_tcga_seed${seed}_M7_EX_RISKHEADDROPONLY_ext.log"
    python -u .\train_light.py --dataset tcga --seed $seed --M7 --rna-genes literature_1500 `
        --external --group-ts $GroupTs 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: M7_EX_RISKHEADDROPONLY seed=$seed" }
    Write-Host "=== M7_EX_RISKHEADDROPONLY seed=$seed Complete: $(Get-Date) ==="
}
Write-Host "=== ALL M7_RISKHEADDROPONLY V2 RUNS COMPLETE: $(Get-Date) ==="
