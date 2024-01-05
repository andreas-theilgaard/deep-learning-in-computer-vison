#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q gpuv100

### -- ask for number of cores (default: 1) -- 
#BSUB -n 4 

### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"

### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"

### -- specify that we need 4GB of memory per core/slot -- 
#BSUB -R "rusage[mem=32GB]"

### -- set walltime limit: hh:mm -- 
#BSUB -W 23:00 

### -- set the email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u s201715@student.dtu.dk
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 

### -- Specify the output and error file. %J is the job-id -- 

### -- -o and -e mean append, -oo and -eo mean overwrite -- 
### -- set the job Name -- 
#BSUB -J project1
#BSUB -o project1.out
#BSUB -e project1.err

nvidia-smi
# Load the cuda module
module load cuda/11.7


source /zhome/8c/c/152745/DLCV/DLCV/bin/activate 

CUDA_VISIBLE_DEVICES=0 python src/hotdog_classifier/test.py