$condaExe = Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"
(& $condaExe "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate PathViT-ray
Set-Location "D:\wonse\Documents\Job\urban_datalab\PATH-ViT"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
Remove-Item Env:\SSL_CERT_FILE -ErrorAction SilentlyContinue

$LogDir = ".logs"
$Seed = 84
$NFolds = 5
$GroupTs = "0807m1pool_uni_kfold5_local_seed84_resume"

# fold0/1은 이전 실행에서 이미 완료됨(kfold_preds CSV 존재 확인) - fold2부터 재개
foreach ($fold in 2,3,4) {
    Write-Host "=== M1_POOL_uni_SS_DISP fold=$fold/$NFolds Start(resume): $(Get-Date) ==="
    $log = Join-Path $LogDir "train_tcga_seed${Seed}_M1_POOL_uni_SS_DISP_kfold5_fold${fold}.log"
    python -u .\train.py --M1_POOL --dataset tcga --external --seed $Seed `
        --backbone uni `
        --patch-keep-frac 0.8 --attn-dispersion `
        --fold $fold --n-folds $NFolds --group-ts $GroupTs 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: M1_POOL_uni_SS_DISP fold=$fold" }
    Write-Host "=== M1_POOL_uni_SS_DISP fold=$fold/$NFolds Complete: $(Get-Date) ==="
}
Write-Host "=== ALL M1_POOL_uni_SS_DISP LOCAL 5-FOLD RUNS COMPLETE(resume): $(Get-Date) ==="
