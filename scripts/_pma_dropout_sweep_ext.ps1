$condaExe = Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"
(& $condaExe "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate PathViT-ray
Set-Location "D:\wonse\Documents\Job\urban_datalab\PATH-ViT"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# findings_backlog.md 13번 항목 후속: risk head 직전 Dropout(0.4) 추가는 negative였는데, 그게
# "위치" 문제인지 "dropout rate 값 자체"가 이 프로젝트 표본 규모에 안 맞는지 구분이 안 됐다.
# cfg.model.dropout(기본 0.3, ViT/ABMIL/RNA/Clinical 인코더 전체 공유)을 --dropout으로 스윕해
# PMA_EX_SS_AUX 기준 단일 시드(42)·external(tcga->cptac)로 빠르게 훑어본다.
# 0.3(기본값)은 기존 PMA_EX_SS_AUX seed42 결과(external C=0.6112)를 그대로 참고값으로 쓴다.

$LogDir = ".logs"
$Dropouts = @(0.1, 0.2, 0.4, 0.5)
$Seed = 42
$GroupTs = "0721pma_dropout_sweep_ext"

$Total = $Dropouts.Count
$Run = 0

foreach ($dropout in $Dropouts) {
    $Run++
    Write-Host "=== [$Run/$Total] PMA_EX_SS_AUX dropout=$dropout seed=$Seed Start: $(Get-Date) ==="
    $log = Join-Path $LogDir "train_tcga_seed${Seed}_PMA_EX_SS_AUX_DROP${dropout}_ext.log"
    python -u .\train.py --dataset tcga --seed $Seed --PMA --rna-genes literature_1500 `
        --patch-keep-frac 0.8 --rna-aux-weight 1.0 --dropout $dropout `
        --external --group-ts $GroupTs 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: PMA_EX_SS_AUX dropout=$dropout" }
    Write-Host "=== [$Run/$Total] PMA_EX_SS_AUX dropout=$dropout seed=$Seed Complete: $(Get-Date) ==="
}
Write-Host "=== ALL PMA_EX_SS_AUX DROPOUT SWEEP RUNS COMPLETE: $(Get-Date) ==="
