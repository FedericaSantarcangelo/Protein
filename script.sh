#!bin/bash -l
#SBATCH --job-name=HDAC6
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --ntasks-per-node=1 #this need to match trainer(devices=...)
#SBATCH --error=err/slurm-%j.err
#SBATCH --output=out/slurm-%j.out
#SBATCH --time=0-1

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate myenv

#args for the script
#srun python -u utils/main.py e tutti gli altri args