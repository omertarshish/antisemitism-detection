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
  - `plots/`: Generated visualizations showing performance trends and error patterns
  - `results/`: Detailed analysis outputs and comparison data

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
