$condaExe = Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"
(& $condaExe "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate PathViT-ray
Set-Location "D:\wonse\Documents\Job\urban_datalab\PATH-ViT"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
Remove-Item Env:\SSL_CERT_FILE -ErrorAction SilentlyContinue

# 레퍼런스 인코더 폭 비율 아이디어(RNA>WSI>Clinical)의 마지막 재시도 - 이번엔 WSI(embed_dim=64)는
# 그대로 두고 RNA=128/Clinical=16만 절대 크기로 축소해서 넣는다(models/vit_pma.py rna_dim/
# clinical_dim, 2026-07-21 2차 추가). 이전 RNA=256/Clinical=16(WSI도 같이 커짐) 시도는 negative -
# WSI 차원 자체를 키운 게 원인이었는지 분리해서 확인.

$LogDir = ".logs"
$Seeds = @(42, 84, 126)
$GroupTs = "0721pma_refdims_ext"

$Total = $Seeds.Count
$Run = 0

foreach ($seed in $Seeds) {
    $Run++
    Write-Host "=== [$Run/$Total] PMA_EX_SS_AUX_RNADIM128_CLINDIM16 seed=$seed Start: $(Get-Date) ==="
    $log = Join-Path $LogDir "train_tcga_seed${seed}_PMA_EX_SS_AUX_refdims_ext.log"
    python -u .\train.py --dataset tcga --seed $seed --PMA --rna-genes literature_1500 `
        --patch-keep-frac 0.8 --rna-aux-weight 1.0 --rna-dim 128 --clinical-dim 16 `
        --external --group-ts $GroupTs 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: PMA_EX_SS_AUX_RNADIM128_CLINDIM16 seed=$seed" }
    Write-Host "=== [$Run/$Total] PMA_EX_SS_AUX_RNADIM128_CLINDIM16 seed=$seed Complete: $(Get-Date) ==="
}
Write-Host "=== ALL PMA_EX_SS_AUX_REFDIMS RUNS COMPLETE: $(Get-Date) ==="
