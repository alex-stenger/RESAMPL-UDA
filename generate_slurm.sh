src_domains=("Domain1" "Domain2" "Domain3")
trg_domains=("Domain1" "Domain2" "Domain3")
models=("segformer")

# Launcher script content
launcher_script_content="#!/bin/bash\n"
launcher_test_content="#!/bin/bash\n"

# Iterate over source and target domains
for src in "${src_domains[@]}"; do
  for trg in "${trg_domains[@]}"; do
    # Skip if source and target are the same
    if [ "$src" != "$trg" ]; then
      for model in "${models[@]}"; do
        run_name="${src}-${trg}_${model}_reti"
        if [ "$model" = "unet_attention" ]; then
          batch_size=3
          img_size=64
        else
          # Set default batch size and image size or perform other actions
          batch_size=5
          img_size=256
        fi
        if [ "$model" = "segformer" ]; then
          lr=0.00006
        else
          lr=0.01
        fi
        if [ "$model" = "ynet" ]; then
          ynet=1
          st=0
        else
          ynet=0
          st=1
        fi
      script_content="#! /bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --exclude=\"hpc-n870\"
#SBATCH --constraint=\"gpua40|gpua100&gpu40g|gpuv100\"
#SBATCH -o jobs/${run_name}.out

hostname
source deactivate
module load python/python-3.11.4
activate deep-learning
python --version
echo 'START'
python3 tools/train.py --network ${model} --run_name ${run_name} --epochs 251 --batch_size ${batch_size} --image_size ${img_size} --dataset_path path_to_src/${src} --dataset_path_trg path_to_trg/${trg} --result_path workdir --self_train ${st} --lr ${lr} --ynet ${ynet} --finetune 0 --pt_ckpt pretrained.pt
echo 'END'
"
          echo "$script_content" > "slurm_scripts/${run_name}.sh"
          launcher_script_content+="sbatch slurm_scripts/${run_name}.sh\n"
      done
    fi
  done
done

# Iterate over source and target domains
for src in "${src_domains[@]}"; do
  for trg in "${trg_domains[@]}"; do
  if [ "$src" != "$trg" ]; then
   for model in "${models[@]}"; do
      
      run_name="${src}-${trg}_${model}_reti"
      if [ "$model" = "unet_attention" ]; then
          batch_size=3
          img_size=64
      else
          # Set default batch size and image size or perform other actions
          batch_size=10
          img_size=256
      fi
      if [ "$model" = "segformer" ]; then
          lr=0.00006
      else
          lr=0.01
      fi
      if [ "$model" = "ynet" ]; then
          st=0
      else
          st=1
      fi
      test_content="#! /bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --constraint=\"gpua40|gpua100|gpuv100|gpurtx6000\"
#SBATCH -o jobs/test_${run_name}.out

hostname
source deactivate
module load python/python-3.11.4
activate deep-learning
python --version
echo 'START'
python3 tools/test.py --network ${model} --batch_size ${batch_size} --image_size ${img_size} --weight_path workdir/${run_name}/ckpt_last.pt --dataset_path path_to_target/${trg} --result_path results/${run_name}_ablation --self_train ${st}
echo 'END'
"

            # Save script to file
            echo "$test_content" > "slurm_scripts/test_${run_name}.sh"
            launcher_test_content+="sbatch slurm_scripts/test_${run_name}.sh\n"
      done
    fi
  done
done

launcher_script_file="launch_train_jobs_reti.sh"
launcher_test_file="launch_test_jobs_reti.sh"
echo -e "$launcher_script_content" > "$launcher_script_file"
echo -e "$launcher_test_content" > "$launcher_test_file"
