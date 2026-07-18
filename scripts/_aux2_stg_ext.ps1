$condaExe = Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"
(& $condaExe "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate PathViT-ray
Set-Location "D:\wonse\Documents\Job\urban_datalab\PATH-ViT"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$LogDir = ".logs"
$Seeds = @(42, 84, 126)
$Datasets = @("tcga", "cptac")
$GroupTs = "0717aux2stg"
$KeepFrac = "0.8"
$AuxWeight = "1.0"
$StageAuxWeight = "1.0"

$Models = @("--M4A", "--PMA")
$Tags   = @("M4A", "PMA")
# Variant A(AUX2만): stage-aux-weight만 추가, clinical 입력은 기존 age/sex 그대로.
# Variant B(AUX2+STG): stage-aux-weight + clinical-staging(ClinicalEncoder에 T/N/M/grade 주입) 둘 다.
$Variants = @(
    @{ Suffix = "AUX2";     ExtraArgs = @() },
    @{ Suffix = "AUX2_STG"; ExtraArgs = @("--clinical-staging") }
)

$Total = $Models.Count * $Variants.Count * $Datasets.Count * $Seeds.Count
$Run = 0

for ($i = 0; $i -lt $Models.Count; $i++) {
    $flag = $Models[$i]; $tag = $Tags[$i]
    foreach ($variant in $Variants) {
        $vsuf = $variant.Suffix
        $vargs = $variant.ExtraArgs
        foreach ($ds in $Datasets) {
            foreach ($seed in $Seeds) {
                $Run++
                Write-Host "=== [$Run/$Total] ${tag}_SS_AUX_${vsuf} ds=$ds seed=$seed Start: $(Get-Date) ==="
                $log = Join-Path $LogDir "train_${ds}_seed${seed}_${tag}_EX_SS_AUX_${vsuf}_ext.log"
                python -u .\train.py --dataset $ds --seed $seed $flag --rna-genes literature_1500 `
                    --patch-keep-frac $KeepFrac --rna-aux-weight $AuxWeight --stage-aux-weight $StageAuxWeight `
                    @vargs --external --group-ts $GroupTs | Tee-Object -FilePath $log
                if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: ${tag}_SS_AUX_${vsuf} ds=$ds seed=$seed" }
                Write-Host "=== [$Run/$Total] ${tag}_SS_AUX_${vsuf} ds=$ds seed=$seed Complete: $(Get-Date) ==="
            }
        }
    }
}
Write-Host "=== ALL AUX2/STG EXTERNAL RUNS COMPLETE: $(Get-Date) ==="
