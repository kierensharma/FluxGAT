# FluxGAT: Integrating Flux Sampling with Graph Neural Networks for Unbiased Gene Essentiality Classification

## Description
FluxGAT introduces a graph neural network (GNN) model designed to predict gene essentiality by utilizing graphical representations derived from flux sampling data. This approach effectively bypasses the traditional reliance on objective functions in flux balance analysis (FBA), thereby reducing observer bias. By harnessing the capabilities of GNNs, FluxGAT adeptly captures the intricate interplay within metabolic reaction networks, facilitating more accurate and unbiased predictions of gene essentiality.

## Installation
To set up your environment for running FluxGAT, please follow these instructions. It is recommended to use a virtual environment to avoid conflicts with existing installations.

First, clone or download this repository to your local machine. Then, navigate to the repository's root directory and run the following command to install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage
To train and evaluate the FluxGAT model, follow the outlined steps sequentially. Make sure to complete each step successfully before moving on to the next.

### Step 1: Flux Sampling
This step involves running the 'flux_sampling.py' script to perform flux sampling on the genome-scale metabolic model (GSMM). The script will generate flux sampling data, which is essential for subsequent steps in the analysis.

**Location:** Begin by navigating to the 'sampling' directory within your project's root folder.

**Command:**
```bash
cd sampling
python flux_sampling.py
```

Expected Output: The script saves the flux sampling data to the data directory. Please verify that the output file(s) are correctly generated in the specified location before proceeding.

## Contributing
Contributions to improve FluxGAT are welcome. If you have suggestions or enhancements, please open an issue first to discuss what you would like to change. For substantial changes, please open a pull request for review.

Ensure to update tests as appropriate and maintain the anonymity of the repository.

## Acknowledgments

This project builds upon the foundational work in flux sampling methodologies and graph neural network architectures. We extend our gratitude to the researchers and developers whose contributions have made this project possible.

## Contact Information
For any inquiries, suggestions, or contributions, please open an issue in this repository, and we will get back to you as soon as possible.