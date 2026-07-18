$condaExe = Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"
(& $condaExe "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate PathViT-ray
Set-Location "D:\wonse\Documents\Job\urban_datalab\PATH-ViT"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# findings_backlog.md 9번 항목 다음 액션 (a): patch-keep-frac은 끄고(기본 1.0) rna-aux-weight만
# 켜서, PMA_EX_SS_AUX의 개선이 RNA aux 단독 효과인지 patch dropout과의 상호작용인지 분리한다.
# (7번 항목: patch dropout 단독으론 null result였으므로, 이 배치가 M4A_EX_SS_AUX/PMA_EX_SS_AUX와
# 비슷하게 나오면 "RNA aux가 주된 기여"라는 기존 결론이 확정되고, 더 낮게 나오면 SS와의 결합이
# 필요했다는 뜻이 된다.)

$LogDir = ".logs"
$Seeds = @(42, 84, 126)
$Datasets = @("tcga", "cptac")
$GroupTs = "0718auxonly"
$AuxWeight = "1.0"

$Models = @("--M4A", "--PMA")
$Tags   = @("M4A", "PMA")

$Total = $Models.Count * $Datasets.Count * $Seeds.Count
$Run = 0

for ($i = 0; $i -lt $Models.Count; $i++) {
    $flag = $Models[$i]; $tag = $Tags[$i]
    foreach ($ds in $Datasets) {
        foreach ($seed in $Seeds) {
            $Run++
            Write-Host "=== [$Run/$Total] ${tag}_EX_AUX ds=$ds seed=$seed Start: $(Get-Date) ==="
            $log = Join-Path $LogDir "train_${ds}_seed${seed}_${tag}_EX_AUX_ext.log"
            python -u .\train.py --dataset $ds --seed $seed $flag --rna-genes literature_1500 `
                --rna-aux-weight $AuxWeight --external --group-ts $GroupTs | Tee-Object -FilePath $log
            if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: ${tag}_EX_AUX ds=$ds seed=$seed" }
            Write-Host "=== [$Run/$Total] ${tag}_EX_AUX ds=$ds seed=$seed Complete: $(Get-Date) ==="
        }
    }
}
Write-Host "=== ALL AUX-ONLY EXTERNAL RUNS COMPLETE: $(Get-Date) ==="
