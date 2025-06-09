target=(MHEALTH HHAR DSA PAMAP2 selfBACK GOTOV)

# target=(selfBACK GOTOV)
for target_index in "${target[@]}"  # Use quotes to handle spaces in values    
do
    sbatch -o logs/complex_classifier_${target_index}.out --account=es_holz --ntasks=1 --cpus-per-task=10 --gpus=rtx_4090:1 --gres=gpumem:7552m --time=300:00:00 --job-name=test --mem-per-cpu=20048 --tmp=120000 --wrap="python -u classifier_bert.py cv v3_v3 ${target_index} 25_125 -s bert_classifier_gru_complex -l 0 -f limu_C24 -c True"
done