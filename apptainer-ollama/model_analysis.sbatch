#!/bin/bash
################################################################################################
### sbatch configuration parameters must start with #SBATCH and must precede any other commands.
################################################################################################
#SBATCH --partition gpu                     # Use GPU partition for guaranteed GPU access
#SBATCH --job-name DeepSeek-V3            # Simplified job name
#SBATCH --output output_logs/job-%J.out  # Output log with job ID
#SBATCH --gpus=rtx_6000:1                  # Request 1 RTX 6000 GPU
#SBATCH --mem=48G                          # max 48 RAM memory allocation

# Set model variables - CORRECTED MODEL NAME
MODEL_NAME="deepseek-r1:7b"           # Use a model already in Ollama
MODEL_SAFE="deepseek-r1:7b"           # Safe directory name
OLLAMA_IP_PORT="[set your ip from the job.out:set the port from job.out"

### Print job information
echo "Starting tweet analysis job with model: ${MODEL_NAME}"
echo "Date: $(date)"
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST
echo -e "GPU allocation:\t\t" $SLURM_JOB_GPUS
echo -e "Ollama endpoint:\t" $OLLAMA_IP_PORT

### Create result directories
mkdir -p results/${MODEL_SAFE}
mkdir -p results/${MODEL_SAFE}/batches

### FIXED Conda activation path
source $HOME/miniconda3/bin/activate tweet-analysis

### Set Python to unbuffered mode for real-time output
export PYTHONUNBUFFERED=TRUE

### Run analysis with model-specific output paths
python cluster_tweet_analysis.py \
  --input dataset_with_full_text.csv \
  --output results/${MODEL_SAFE}/analysis_full.csv \
  --ip-port "${OLLAMA_IP_PORT}" \
  --model "${MODEL_NAME}" \
  --workers 4 \
  --batch-size 50
  
echo "Job completed at $(date)"