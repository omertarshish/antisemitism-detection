# Antisemitism Detection in Social Media

A research project evaluating large language models (LLMs) in detecting antisemitic content according to formal definitions.

## Project Overview

This project investigates how Language Models of different sizes (DeepSeek-R1 7B and 8B) perform in identifying antisemitic content across social media posts. The analysis compares model performance using two different definition frameworks:

1. **International Holocaust Remembrance Alliance (IHRA)** definition of antisemitism
2. **Jerusalem Declaration on Antisemitism (JDA)** definition

The project aims to evaluate how model size affects performance metrics, with a special focus on reducing false positives while maintaining detection capabilities.

## Key Components

### Data Processing Pipeline

- **Data Collection**: Uses a dataset of real social media posts with manually labeled antisemitism tags
- **Cluster Processing**: Leverages HPC cluster resources for efficient parallel processing of the dataset
- **Model Integration**: Utilizes Ollama containerized through Apptainer for inference with different model sizes

### Analysis Framework

- **Performance Metrics**: Focused on precision, specificity, false positive rates, and F-scores
- **Error Analysis**: Deep analysis of false positives by tweet length, keywords, and content topics
- **Topic Modeling**: Analysis of false positive patterns revealing common term clusters (e.g., "jews https israel") that trigger misclassifications
- **Model Comparison**: Systematic comparison between 7B and 8B model performance
- **Visualization**: Comprehensive plots and charts to illustrate findings

## Repository Structure

- **apptainer-ollama/**: Scripts and configuration for running models on HPC clusters
  - `cluster_tweet_analysis.py`: Main script for processing tweets with LLMs
  - `model_analysis.sbatch`: SLURM batch script for cluster execution
  - `results/`: Raw model outputs and batch processing results

- **analyze results/**: Tools for comparative analysis and visualization
  - `analyze_results.py`: Script for generating performance metrics and visualizations
  - `merge_results.py`: Script for combining different model results
  - `plots/`: Generated visualizations showing performance trends and error patterns
  - `results/`: Detailed analysis outputs and comparison data

- **analyze_results_colab.ipynb**: Jupyter notebook for interactive analysis and visualization

## Key Findings

- The 8B model shows improved precision compared to the 7B model, but with substantially higher false positive rates
- Length of text significantly impacts false positive rates, with longer texts showing different error patterns
- Topic modeling reveals common themes in false positive cases, with terms like "jews," "https," and "israel" frequently appearing together in false positives across all models
- False positive analysis shows that specific combinations of terms like "jews https amp" and "jews https israel" consistently trigger misclassifications
- IHRA and JDA definitions result in different error patterns, providing insights for detection framework design

## Installation & Usage

### Prerequisites

- Python 3.8+
- Pandas, NumPy, Matplotlib, Seaborn, scikit-learn
- For cluster execution: Apptainer, Ollama, SLURM scheduler

### Running the Analysis Pipeline

1. **Set up the environment**:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn tqdm requests
   ```

2. **Process dataset with models**:
   ```bash
   cd apptainer-ollama
   sbatch ollama.sbatch  # For cluster execution
   
   # For local execution:
   python cluster_tweet_analysis.py --input dataset_with_full_text.csv --output results/analysis_output.csv --ip-port localhost:11434 --model deepseek-r1:7b
   ```

3. **Analyze results**:
   ```bash
   cd ../analyze\ results/
   python analyze_results.py --input comparative_model_results.csv --fp-analysis --topics 5
   ```

4. **View generated reports and visualizations**:
   - Check `plots/` directory for performance visualizations
   - Examine `results/` directory for detailed metrics and comparisons

## Complete Guide for Using Ollama with Apptainer on the Cluster

This guide will help you set up and run Ollama using Apptainer on the cluster and use it to analyze tweets with the `cluster_tweet_analysis.py` script.

### 1. Create the Working Directory

```bash
mkdir -p ~/apptainer-ollama
cd ~/apptainer-ollama
```

### 2. Prepare the Files

You need these key files in your directory:
- `apptainer-ollama.sh`: Script to build and run Ollama in Apptainer
- `ollama.sbatch`: Script to submit the Ollama job to the cluster
- `cluster_tweet_analysis.py`: Python script for tweet analysis
- `model_analysis.sbatch`: Script to run the analysis job
- `Updated_dataset_with_full_text.csv`: Your dataset

Make sure all these files are uploaded to your `apptainer-ollama` directory.

### 3. Starting the Ollama Server

1. First, submit the job to start the Ollama server:

```bash
cd ~/apptainer-ollama
sbatch ollama.sbatch
```

2. Check the output file to find the IP and PORT:

```bash
cat job-*.out
```

3. Look for a line like this in the output:
```
PORT: 9988
Server IP: 132.72.66.187
```

Note down this IP and PORT - you'll need it for the next step.

### 4. Running the Tweet Analysis

1. Edit the `model_analysis.sbatch` file to update these settings:

```bash
# Open the file
nano model_analysis.sbatch
```

2. Update the following key settings:

```bash
# Choose the GPU type you need
#SBATCH --gpus=rtx_6000:1

# Set the model you want to use
MODEL_NAME="llama3:8b"

# Set a directory name for results (avoid using colons)
MODEL_SAFE="llama3-8b"

# IMPORTANT: Update with the IP:PORT from the ollama.sbatch output
OLLAMA_IP_PORT="132.72.66.187:9988"
```

3. Make sure the output directories exist:

```bash
mkdir -p output_logs
mkdir -p results
```

4. Submit the analysis job:

```bash
sbatch model_analysis.sbatch
```

5. Monitor the job:

```bash
# Check the job status
squeue -u $USER

# View the job output in real-time
tail -f output_logs/job-*.out
```

### 5. Understanding the Results

The analysis results will be saved in:
- `results/MODEL_SAFE/analysis_full.csv`: Complete results
- `results/MODEL_SAFE/batches/`: Individual batch results

Each CSV file contains the original tweet data plus:
- `IHRA_Decision`: Whether the tweet is antisemitic per IHRA definition
- `IHRA_Explanation`: Explanation for the IHRA decision
- `JDA_Decision`: Whether the tweet is antisemitic per JDA definition
- `JDA_Explanation`: Explanation for the JDA decision

### 6. Troubleshooting

- **If the Ollama server fails**: Check if the port is already in use or if you have GPU access
- **If the analysis job fails**: Check if the IP:PORT is correct and if Ollama is running
- **For conda activation issues**: Make sure you have the required environment with pandas, requests, and tqdm installed

### 7. Example: Complete Workflow

```bash
# 1. Create directory and navigate to it
mkdir -p ~/apptainer-ollama
cd ~/apptainer-ollama

# 2. Start Ollama server
sbatch ollama.sbatch

# 3. Check the output to get the IP:PORT
cat job-*.out
# Note down the IP:PORT (e.g., 132.72.66.187:9988)

# 4. Edit model_analysis.sbatch with the correct IP:PORT
nano model_analysis.sbatch
# Update OLLAMA_IP_PORT="132.72.66.187:9988"

# 5. Create output directories
mkdir -p output_logs results

# 6. Submit the analysis job
sbatch model_analysis.sbatch

# 7. Monitor the job
tail -f output_logs/job-*.out
```

Your results will be available in the `results` directory once the job completes.

## Contributing

This is a research project exploring antisemitism detection with LLMs. If you're interested in contributing or building upon this work, please:

1. Review the methodology in the paper (see `Antisemitism_Detection_in_Social_Media.pdf`)
2. Consider both technical improvements and ethical implications
3. Contact the repository maintainers for collaboration opportunities

## License

This research code is shared for academic and research purposes.

## Acknowledgments

- Research conducted using DeepSeek-R1 open source models
- Data processing leveraged high-performance computing resources
- Analysis framework builds on established metrics for classification evaluation