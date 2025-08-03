# U-Net with OSA for Spectroscopic Analysis

This repository contains the implementation and data accompanying the research described in our paper, available as a preprint at [SSRN](https://ssrn.com/abstract=5128894).

The U-Net architecture combined with Optical Spectrum Analysis (OSA) techniques is applied for spectroscopic analysis, specifically focusing on accurately simulating and analyzing isotopic abundance from spectroscopic data. In cases where strong self-reversal effects occur, the center wavelength was determined from the absorption profile.

## Project Structure

```
U-net_with_OSA/
├── data/
├── weights/
├── .gitignore
├── datarun_for_isotope.py
├── model_utils.py
├── test_model.py
├── train_model.py
├── VoigtClass.py
├── requirements.txt
└── LICENSE
```

## File Descriptions

- `data/`: Contains the spectroscopic datasets demonstrating clear self-reversal effects as presented in the paper.
- `weights/`: Includes the trained model weights split into parts due to file size constraints. You must decompress and merge these files to use the trained model.
- `model_utils.py`: Contains the U-Net model architecture used in the research.
- `VoigtClass.py`: Provides computational tools for calculating Voigt profiles essential for spectroscopic analysis.
- `datarun_for_isotope.py`: Loads and processes measurement data, enabling the reproduction of the analysis discussed in the paper.
- `train_model.py`: Script for training the U-Net model using the provided dataset. Adjustments for computational resources are also included.
- `test_model.py`: Allows evaluation of the trained model on simulation data, assessing model accuracy and performance.

## Usage

### Setup Environment

First, create the required Python environment using:

```bash
pip install -r requirements.txt
```

### Running the Analysis

To reproduce the spectroscopic data analysis presented in the paper, execute:

```bash
python datarun_for_isotope.py
```

### Training the Model

To train the U-Net model with the provided dataset, execute:

```bash
python train_model.py
```

#### Adjusting for Computational Resources

If encountering memory or resource limitations during training, modify the `num_samples` parameter in `train_model.py` to a smaller value. This reduces the dataset size, allowing training to proceed with fewer computational resources:

```python
num_samples = 100000  # Adjust this number as needed
```

### Evaluating the Model

After preparing the trained weights in the `weights/` folder, you can evaluate the model's performance using:

```bash
python test_model.py
```

Ensure that the weights (`*.pth`) are properly combined after decompression for accurate evaluation.

## Reference

For detailed methodologies, results, and discussions, please refer to our paper available at:

- [Preprint on SSRN](https://doi.org/10.1016/j.rinp.2025.108373)

## License

This project is distributed under the terms specified in the `LICENSE` file.

