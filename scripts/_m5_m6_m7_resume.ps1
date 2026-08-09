$condaExe = Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"
(& $condaExe "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate PathViT-ray
Set-Location "D:\wonse\Documents\Job\urban_datalab\PATH-ViT"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
Remove-Item Env:\SSL_CERT_FILE -ErrorAction SilentlyContinue

# m5_m6_m7_multiseed_kfold_local.ps1이 19:03경 죽음(원인 불명, 같은 세션에서 M1_POOL 때와 동일
# 패턴) - M5는 15/15 완료돼 있으니 재실행 안 함. M6 seed42 fold0-2까지 끝나 있으니 fold3부터 재개.

$LogDir = ".logs"
$NFolds = 5
$GroupTs = "0808_m5m6m7_multiseed_kfold5_local_resume"

$M6Args = @("--M6", "--rna-genes", "literature_1500_intersection")
$M7Args = @("--M7", "--rna-genes", "literature_1500_intersection", "--clinical-margin", "--clinical-staging", "--combine-mode", "cox_add")

# M6 나머지: seed42 fold3-4, seed84/126 전체
$M6Remaining = @(
    @{ Seed = 42; Folds = @(3, 4) },
    @{ Seed = 84; Folds = @(0, 1, 2, 3, 4) },
    @{ Seed = 126; Folds = @(0, 1, 2, 3, 4) }
)
foreach ($item in $M6Remaining) {
    foreach ($fold in $item.Folds) {
        $seed = $item.Seed
        Write-Host "=== M6 seed=$seed fold=$fold/$NFolds Start: $(Get-Date) ==="
        $log = Join-Path $LogDir "train_tcga_seed${seed}_M6_kfold5_fold${fold}.log"
        $fullArgs = $M6Args + @("--dataset", "tcga", "--external", "--seed", "$seed", "--fold", "$fold", "--n-folds", "$NFolds", "--group-ts", $GroupTs)
        python -u .\train_light.py @fullArgs 2>&1 | Tee-Object -FilePath $log
        if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: M6 seed=$seed fold=$fold" }
        Write-Host "=== M6 seed=$seed fold=$fold/$NFolds Complete: $(Get-Date) ==="
    }
}

# M7 전체(3seed x 5fold)
foreach ($seed in @(42, 84, 126)) {
    for ($fold = 0; $fold -lt $NFolds; $fold++) {
        Write-Host "=== M7 seed=$seed fold=$fold/$NFolds Start: $(Get-Date) ==="
        $log = Join-Path $LogDir "train_tcga_seed${seed}_M7_kfold5_fold${fold}.log"
        $fullArgs = $M7Args + @("--dataset", "tcga", "--external", "--seed", "$seed", "--fold", "$fold", "--n-folds", "$NFolds", "--group-ts", $GroupTs)
        python -u .\train_light.py @fullArgs 2>&1 | Tee-Object -FilePath $log
        if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: M7 seed=$seed fold=$fold" }
        Write-Host "=== M7 seed=$seed fold=$fold/$NFolds Complete: $(Get-Date) ==="
    }
}
Write-Host "=== ALL M5/M6/M7 LOCAL MULTISEED KFOLD RUNS COMPLETE(resume): $(Get-Date) ==="
