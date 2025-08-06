#!/bin/bash
#SBATCH --time=48:00:00 # Run time
#SBATCH --nodes 1  # Number of reaquested nodes 
#SBATCH --ntasks-per-node=1
#SBATCH --mem 400G
#SBATCH -c 30
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100|H100.80gb"
#SBTACH --job-name U_net_after_sweep_on_MT_march_25
#SBATCH --error=U_net__after_sweep__on_MT_march_25_error.o%j
#SBATCH --output=U_net__after_sweep__on_MT_march_25_output.o%j
#SBATCH --requeue
#SBATCH --mail-user=asarker@uni-osnabrueck.de

#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END 
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
##SBATCH --mail-type=ALL

#SBATCH --signal=SIGTERM@90
echo "running in shell: " "$SHELL"

export NCCL_SOCKET_IFNAME=lo

## to force NCCL to use share memory and not infiniband
##export NCCL_IB_DISABLE=1

## 
export TMPDIR='/share/klab/argha' 

## Please add any modules you want to load here, as an example we have commented out the modules
## that you may need such as cuda, cudnn, miniconda3, uncomment them if that is your use case 
## term handler the function is executed once the job gets the TERM signal

spack load miniconda3

eval "$(conda shell.bash hook)"
conda activate llm

###### Target ins to run 2 experiments with 2 different data sets

#### 1 with kernal size 3 and augmentation



### 2nd is kernal size 9,3,3 and augmentation



### 3rd is kernal size 7,5,5

## srun python u_net_test_experiments_mixed_data_sweep.py
##srun python u_net_test_experiments_mixed_data_kernal_3.py
srun python flan-T5-base.py
srun python fine_tuned_flan-T5.py
srun python peft_fine_tuned.py
srun python evaluation.py


##srun python u_net_test_experiments_mixed_data_kernal_3_experiment_7_5_5.py


