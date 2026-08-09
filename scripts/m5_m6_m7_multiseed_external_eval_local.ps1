$condaExe = Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"
(& $condaExe "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate PathViT-ray
Set-Location "D:\wonse\Documents\Job\urban_datalab\PATH-ViT"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
Remove-Item Env:\SSL_CERT_FILE -ErrorAction SilentlyContinue

# m5_m6_m7_multiseed_kfold_local.ps1(45개 학습)이 저장해 둔 checkpoint를 재학습 없이 다시 불러와
# train_light.py --eval-external-ckpt(2026-08-08 추가)로 external(cptac) 예측만 재추출해
# .logs/external_preds/에 CSV로 저장한다. sbatch/pma_uni2_..._multiseed_external_eval_hpc.sh와
# 동일한 관례(checkpoint를 seed+fold로 좁혀서 와일드카드로 찾음).
#
# 완료 후 internal+external 한 번에: python -m scripts.summarize_multiseed_all_models

$Seeds = @(42, 84, 126)
$NFolds = 5

$Models = @(
    @{ Name = "M5"; Glob = "survival_tcga_best_m5*"; Args = @("--M5", "--clinical-margin", "--clinical-staging") },
    @{ Name = "M6"; Glob = "survival_tcga_best_m6*"; Args = @("--M6", "--rna-genes", "literature_1500_intersection") },
    @{ Name = "M7"; Glob = "survival_tcga_best_m7*"; Args = @("--M7", "--rna-genes", "literature_1500_intersection", "--clinical-margin", "--clinical-staging", "--combine-mode", "cox_add") }
)

foreach ($m in $Models) {
    foreach ($seed in $Seeds) {
        for ($fold = 0; $fold -lt $NFolds; $fold++) {
            $pattern = "$($m.Glob)fold${fold}of${NFolds}_seed${seed}_light.pt"
            $matches = Get-ChildItem -Path "models/checkpoint" -Filter $pattern -ErrorAction SilentlyContinue
            if (-not $matches) {
                Write-Host "[SKIP] $($m.Name) seed=$seed fold=$fold: checkpoint 못 찾음 (패턴: $pattern)"
                continue
            }
            if ($matches.Count -gt 1) {
                Write-Host "[경고] $($m.Name) seed=$seed fold=$fold: checkpoint가 $($matches.Count)개 매칭됨 — 첫 번째만 사용: $($matches[0].Name)"
            }
            $ckpt = "models/checkpoint/$($matches[0].Name)"

            Write-Host "=== external eval-only: $($m.Name) seed=$seed fold=$fold ckpt=$ckpt Start: $(Get-Date) ==="
            $fullArgs = $m.Args + @(
                "--dataset", "tcga", "--external", "--seed", "$seed",
                "--fold", "$fold", "--n-folds", "$NFolds", "--eval-external-ckpt", $ckpt
            )
            python -u .\train_light.py @fullArgs
            Write-Host "=== external eval-only: $($m.Name) seed=$seed fold=$fold Complete: $(Get-Date) ==="
        }
    }
}
Write-Host "=== ALL M5/M6/M7 LOCAL EXTERNAL EVAL RUNS COMPLETE: $(Get-Date) ==="
