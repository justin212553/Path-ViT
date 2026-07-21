$condaExe = Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"
(& $condaExe "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate PathViT-ray
Set-Location "D:\wonse\Documents\Job\urban_datalab\PATH-ViT"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# CPTAC 재타일링(data/patches_cptac_512)이 tiling만 되고 feature 추출이 안 돼 있었던 것을 확인함
# (features.pt 0개) - .done 마커가 있는 564개 슬라이드는 재타일링 없이 스킵되고 feature 추출만
# 이어서 진행된다.
Write-Host "=== CPTAC 재타일링 feature 추출 시작: $(Get-Date) ==="
python -u -m data.preprocess --dataset cptac --output-dir data/patches_cptac_512 2>&1 | Tee-Object -FilePath ".logs\cptac_preprocess_512_features.log"
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: CPTAC feature 추출" }
Write-Host "=== CPTAC 재타일링 feature 추출 완료: $(Get-Date) ==="

$featCount = (Get-ChildItem -Path "data\patches_cptac_512\tiles" -Recurse -Filter "features.pt" -ErrorAction SilentlyContinue).Count
Write-Host "CPTAC features.pt 개수: $featCount"

# findings_backlog.md 4번 항목 핵심 검증 - 재타일링(512px@0.5MPP) 데이터로 지금까지 external 최고
# 기록인 M4A_EX_SS_AUX/PMA_EX_SS_AUX를 정식으로(3시드 x tcga/cptac 양방향) 재검증해, M7_EX(기존
# 타일링 기준 external C=0.634)와의 격차가 좁혀지는지 확인한다.
$LogDir = ".logs"
$Seeds = @(42, 84, 126)
$Datasets = @("tcga", "cptac")
$GroupTs = "0719retiled_ssaux"

$Models = @("--M4A", "--PMA")
$Tags   = @("M4A", "PMA")

$Total = $Models.Count * $Datasets.Count * $Seeds.Count
$Run = 0

for ($i = 0; $i -lt $Models.Count; $i++) {
    $flag = $Models[$i]; $tag = $Tags[$i]
    foreach ($ds in $Datasets) {
        foreach ($seed in $Seeds) {
            $Run++
            Write-Host "=== [$Run/$Total] ${tag}_EX_SS_AUX(retiled) ds=$ds seed=$seed Start: $(Get-Date) ==="
            $log = Join-Path $LogDir "train_${ds}_seed${seed}_${tag}_EX_SS_AUX_retiled_ext.log"
            python -u .\train.py --dataset $ds --seed $seed $flag --rna-genes literature_1500 `
                --patch-keep-frac 0.8 --rna-aux-weight 1.0 `
                --patches-root-tcga data/patches_tcga_512 --patches-root-cptac data/patches_cptac_512 `
                --external --group-ts $GroupTs 2>&1 | Tee-Object -FilePath $log
            if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: ${tag}_EX_SS_AUX(retiled) ds=$ds seed=$seed" }
            Write-Host "=== [$Run/$Total] ${tag}_EX_SS_AUX(retiled) ds=$ds seed=$seed Complete: $(Get-Date) ==="
        }
    }
}
Write-Host "=== ALL RETILED SS_AUX EXTERNAL RUNS COMPLETE: $(Get-Date) ==="
