set -e
mkdir -p /c/porpoise_paad_amil /c/porpoise_paad_snn
echo "=== AMIL (WSI only) ==="
python main.py --which_splits 5foldcv --split_dir tcga_paad --data_root_dir ./inputs --mode path --model_type porpoise_amil --path_input_dim 1536 --results_dir C:/porpoise_paad_amil
echo "=== SNN (genomic only) ==="
python main.py --which_splits 5foldcv --split_dir tcga_paad --data_root_dir ./inputs --mode omic --reg_type omic --model_type snn --results_dir C:/porpoise_paad_snn
