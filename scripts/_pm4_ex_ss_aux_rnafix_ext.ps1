$condaExe = Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"
(& $condaExe "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate PathViT-ray
Set-Location "D:\wonse\Documents\Job\urban_datalab\PATH-ViT"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# 레퍼런스 M4의 RNA 결합 방식(풀링 후 post-hoc sigmoid 게이트)이 사실 M4A/PMA(co-attention)보다
# PM4(다성분 풀링 + post-hoc 게이트)에 구조적으로 더 가깝다는 걸 확인했다. RNA 전처리 버그 수정
# 이후 PM4_EX_SS_AUX(literature_1500 + patch dropout 0.8 + RNA aux 1.0)로 재검증.
# external은 tcga train -> cptac test 단일 방향, 3시드.

$LogDir = ".logs"
$Seeds = @(42, 84, 126)
$GroupTs = "0721pm4_ex_ss_aux_rnafix_ext"

$Total = $Seeds.Count
$Run = 0

foreach ($seed in $Seeds) {
    $Run++
    Write-Host "=== [$Run/$Total] PM4_EX_SS_AUX_RNAFIX seed=$seed Start: $(Get-Date) ==="
    $log = Join-Path $LogDir "train_tcga_seed${seed}_PM4_EX_SS_AUX_RNAFIX_ext.log"
    python -u .\train.py --dataset tcga --seed $seed --PM4 --rna-genes literature_1500 `
        --patch-keep-frac 0.8 --rna-aux-weight 1.0 `
        --external --group-ts $GroupTs 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: PM4_EX_SS_AUX_RNAFIX seed=$seed" }
    Write-Host "=== [$Run/$Total] PM4_EX_SS_AUX_RNAFIX seed=$seed Complete: $(Get-Date) ==="
}
Write-Host "=== ALL PM4_EX_SS_AUX_RNAFIX EXTERNAL RUNS COMPLETE: $(Get-Date) ==="
