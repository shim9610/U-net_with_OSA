# U-Net with OSA for Spectroscopic Analysis

This repository contains the implementation and accompanying data for the research described in our paper, available as a preprint at [SSRN](https://ssrn.com/abstract=5128894).

We employ a modified U-Net architecture incorporating One-Shot Aggregation (OSA) from VoVNet, specifically tailored for spectroscopic analysis. Our approach addresses self-reversal and self-absorption effects in laser-induced breakdown spectroscopy (LIBS), enabling accurate reconstruction of isotopic abundance from spectroscopic data.

## Project Structure

```
U-net_with_OSA/
├── data/                      # Spectroscopic datasets demonstrating self-reversal
├── weights/                   # Trained model weights (split due to file size constraints)
├── .gitignore
├── datarun_for_isotope.py     # Processes measurement data and reproduces analysis
├── model_utils.py             # Defines U-Net model architecture
├── test_model.py              # Evaluates the trained model on simulation data
├── VoigtClass.py              # Computes Voigt profiles for spectroscopic analysis
├── requirements.txt           # Python dependencies
└── LICENSE                    # Licensing information
```

## Installation

First, ensure Python 3.8 or higher is installed. Then set up the environment:

```bash
pip install -r requirements.txt
```

If GPU support is needed, ensure CUDA-compatible PyTorch is installed.

## Usage
### Model Evaluation

After decompressing and combining the model weights into the `weights/` directory, evaluate model performance using:

```bash
python test_model.py
```

Ensure the model weights (`*.pth`) are correctly merged before evaluation.

### Data Analysis

Reproduce the spectroscopic data analysis:

```bash
python datarun_for_isotope.py
```

## Notes on Resources

The model provided is resource-intensive and requires a GPU with significant memory (e.g., NVIDIA RTX 4090 recommended). If you experience resource constraints:

- Reduce batch size in `test_model.py` or training scripts.
- Adjust model channels in `model_utils.py` to lower computational requirements.

Detailed guidance is provided in comments within the scripts.

## Citation

Please cite our research as follows:

[Your detailed citation information here.]

## License

Distributed under the terms specified in the `LICENSE` file.

