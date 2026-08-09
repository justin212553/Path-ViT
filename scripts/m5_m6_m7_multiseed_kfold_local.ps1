$condaExe = Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"
(& $condaExe "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate PathViT-ray
Set-Location "D:\wonse\Documents\Job\urban_datalab\PATH-ViT"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
Remove-Item Env:\SSL_CERT_FILE -ErrorAction SilentlyContinue

# M5(Clinical, margin+staging)/M6(RNA, INT1500)/M7(Clinical+RNA, margin+staging, cox_add)를
# 3seed(42/84/126) x 5-fold = 45개 run으로 로컬에서 학습(train_light.py, WSI 없어서 fold당
# 수십초~수분 수준으로 빠름). M5/M7은 PMA와 동일한 정보량(margin+staging)으로 맞춰야 공정한
# 비교가 되고, M7은 cox_add가 R_ONLY(margin만)의 internal을 크게 끌어올렸던 전례가 있어 combine
# -mode도 PMA와 통일한다. M1_POOL/M2_POOL/M3(UNI2)는 HPC에서 별도로 돈다
# (sbatch/m{1,2}_pool_uni2_multiseed_kfold_array_hpc.sh, sbatch/m3_uni2_multiseed_kfold_array_hpc.sh).
#
# 완료 후 internal+external 한 번에: python -m scripts.summarize_multiseed_all_models
# (이 시점엔 M5/M6/M7만 internal 나오고, external은 m5_m6_m7_multiseed_external_eval_local.ps1
# 까지 돌려야 채워짐)

$LogDir = ".logs"
$Seeds = @(42, 84, 126)
$NFolds = 5
$GroupTs = "0808_m5m6m7_multiseed_kfold5_local"

$Models = @(
    @{ Name = "M5"; Args = @("--M5", "--clinical-margin", "--clinical-staging") },
    @{ Name = "M6"; Args = @("--M6", "--rna-genes", "literature_1500_intersection") },
    @{ Name = "M7"; Args = @("--M7", "--rna-genes", "literature_1500_intersection", "--clinical-margin", "--clinical-staging", "--combine-mode", "cox_add") }
)

foreach ($m in $Models) {
    foreach ($seed in $Seeds) {
        for ($fold = 0; $fold -lt $NFolds; $fold++) {
            Write-Host "=== $($m.Name) seed=$seed fold=$fold/$NFolds Start: $(Get-Date) ==="
            $log = Join-Path $LogDir "train_tcga_seed${seed}_$($m.Name)_kfold5_fold${fold}.log"
            $fullArgs = $m.Args + @(
                "--dataset", "tcga", "--external", "--seed", "$seed",
                "--fold", "$fold", "--n-folds", "$NFolds", "--group-ts", $GroupTs
            )
            python -u .\train_light.py @fullArgs 2>&1 | Tee-Object -FilePath $log
            if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $($m.Name) seed=$seed fold=$fold" }
            Write-Host "=== $($m.Name) seed=$seed fold=$fold/$NFolds Complete: $(Get-Date) ==="
        }
    }
}
Write-Host "=== ALL M5/M6/M7 LOCAL MULTISEED KFOLD RUNS COMPLETE: $(Get-Date) ==="
