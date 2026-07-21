$condaExe = Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"
(& $condaExe "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate PathViT-ray
Set-Location "D:\wonse\Documents\Job\urban_datalab\PATH-ViT"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# findings_backlog.md 13번 항목 후속: 단일 시드(42) dropout 스윕에서 0.3만 빼고 전부 붕괴하는
# 부자연스러운 패턴이 나왔다 - 시드 편차 아티팩트인지 확인하기 위해 0.2/0.4를 seed 84/126으로
# 추가 검증한다(seed42는 이미 완료, PMA_EX_SS_AUX 기준, tcga->cptac external).

$LogDir = ".logs"
$Dropouts = @(0.2, 0.4)
$Seeds = @(84, 126)
$GroupTs = "0721pma_dropout0204_multiseed_ext"

$Total = $Dropouts.Count * $Seeds.Count
$Run = 0

foreach ($dropout in $Dropouts) {
    foreach ($seed in $Seeds) {
        $Run++
        Write-Host "=== [$Run/$Total] PMA_EX_SS_AUX dropout=$dropout seed=$seed Start: $(Get-Date) ==="
        $log = Join-Path $LogDir "train_tcga_seed${seed}_PMA_EX_SS_AUX_DROP${dropout}_ext.log"
        python -u .\train.py --dataset tcga --seed $seed --PMA --rna-genes literature_1500 `
            --patch-keep-frac 0.8 --rna-aux-weight 1.0 --dropout $dropout `
            --external --group-ts $GroupTs 2>&1 | Tee-Object -FilePath $log
        if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: PMA_EX_SS_AUX dropout=$dropout seed=$seed" }
        Write-Host "=== [$Run/$Total] PMA_EX_SS_AUX dropout=$dropout seed=$seed Complete: $(Get-Date) ==="
    }
}
Write-Host "=== ALL PMA_EX_SS_AUX DROPOUT 0.2/0.4 MULTISEED RUNS COMPLETE: $(Get-Date) ==="
