#!/bin/bash

#SBATCH -J llama-labelling
#SBATCH -p gpu
#SBATCH -A r00286
#SBATCH -o ./logs/%j.txt
#SBATCH -e ./logs/%j.err
#SBATCH --mail-type=ALL
#SBATCH --mem=64G
#SBATCH --mail-user=demistry@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00

# print which SHELL
# echo "SHELL: ${SHELL}"


# follow instructions in the readme and then create the venv, activate it
# source ../llm.c/.venv/bin/activate ## CHANGE
echo "python version: $(python --version)"
echo "Python: $(which python)"

python -c "print('*'*50)"

num_gpus=$(python -c "import torch; print(torch.cuda.device_count())")

python process-infant-images/label_script.py \
    --hf_token <generate-your-own-hf-token> \
    --input_path '/N/project/infant_image_statistics/video_frames/046TE' \
    --output_path './outputs' \
    --num_gpus $num_gpus \
    --corrupt_folder ./corrupt/

python -c "print('*'*50)"
echo "Done"
