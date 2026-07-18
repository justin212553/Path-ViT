$condaExe = Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"
(& $condaExe "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate PathViT-ray
Set-Location "D:\wonse\Documents\Job\urban_datalab\PATH-ViT"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# findings_backlog.md 3번 항목: train_light.py의 lr=1e-3 기본값이 스모크 테스트에서 M6를
# train_c_index 0.99까지 과적합시키는 게 확인됐다. 지금 M7_EX(external C=0.634, p=0.0025)를
# 이 프로젝트의 baseline으로 삼고 있으므로, WSI 모델과 동일한 lr=1e-5로 M5/M6/M6X/M7을
# EX(literature_1500) 모드로 재검증해 baseline 수치 자체를 더 단단하게 잡는다.
# (아키텍처가 아직 확정 안 된 상태라 본격 lr 스윕은 미룸 - 이건 baseline 재확인용 단발성 실행.)

$LogDir = ".logs"
$Seeds = @(42, 84, 126)
$Datasets = @("tcga", "cptac")
$GroupTs = "0717lightlr1e5"
$Lr = "1e-5"

$Models = @("--M5", "--M6", "--M6X", "--M7")
$Tags   = @("M5", "M6", "M6X", "M7")

$Total = $Models.Count * $Datasets.Count * $Seeds.Count
$Run = 0

for ($i = 0; $i -lt $Models.Count; $i++) {
    $flag = $Models[$i]; $tag = $Tags[$i]
    foreach ($ds in $Datasets) {
        foreach ($seed in $Seeds) {
            $Run++
            Write-Host "=== [$Run/$Total] ${tag}_EX_LR1e-05 ds=$ds seed=$seed Start: $(Get-Date) ==="
            $log = Join-Path $LogDir "train_light_${ds}_seed${seed}_${tag}_EX_LR1e-05_ext.log"
            python -u .\train_light.py --dataset $ds --seed $seed $flag --rna-genes literature_1500 `
                --lr $Lr --external --group-ts $GroupTs | Tee-Object -FilePath $log
            if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: ${tag}_EX_LR1e-05 ds=$ds seed=$seed" }
            Write-Host "=== [$Run/$Total] ${tag}_EX_LR1e-05 ds=$ds seed=$seed Complete: $(Get-Date) ==="
        }
    }
}
Write-Host "=== ALL LIGHT EX LR1e-05 EXTERNAL RUNS COMPLETE: $(Get-Date) ==="
