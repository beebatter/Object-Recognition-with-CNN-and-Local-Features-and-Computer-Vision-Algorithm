# Object Recognition with CNN and Local Features

**Author**: Kunwei Song
**Student ID**: 11537231

This repository contains experiments comparing convolutional neural networks (CNNs) and traditional computer-vision (CV) pipelines on the iCubWorld1.0 and CIFAR-10 datasets. It supports end-to-end training and evaluation of lightweight CNN architectures as well as a Bag-of-Visual-Words (BoVW) approach using SIFT features and SVM classifiers.

## Repository Structure

```bash
├── Dataset/
│   └── iCubWorld1/            # iCubWorld1.0 dataset folder
│       ├── train/             # Training images (3 instances × 200 images per category)
│       └── test/              # Test subsets: Background, Categorization, Demonstrator, Robot
├── iCubWorld1_CNN_Improved.ipynb          # CNN experiments on iCubWorld1.0 dataset (improved)
├── iCubWorld_Traditional_CV_Improved.ipynb # BoVW + SVM pipeline on iCubWorld1.0 dataset (improved)
├── Cifar_10_Tradition_CV.ipynb            # Traditional CV pipeline experiments on CIFAR-10
└── requirements.txt            # Python dependencies
```

## Dependencies

The code was developed and tested with Python 3.8+. Install the required packages:

```bash
pip install -r requirements.txt
```

Typical dependencies include:

* numpy
* opencv-python
* scikit-learn
* matplotlib
* tqdm
* torch
* torchvision
* jupyter

## Usage

### 1. Prepare the Dataset

1. Download and extract the iCubWorld1.0 dataset into `Dataset/iCubWorld1/`, preserving the `train/` and `test/` subdirectories.
2. CIFAR-10 is fetched automatically via `fetch_openml` in the notebooks, so no manual download is needed.

### 2. Launch Jupyter Notebooks

```bash
jupyter notebook
```

Open and run the notebooks in the following order:

1. **icubworld1\_cnn\_experiment.ipynb**: trains and evaluates a custom CNN and an improved global-pooling CNN on the iCubWorld1.0 dataset.
2. **traditional\_CV\_\_Cifar\_10.ipynb**: implements a SIFT + BoVW + SVM pipeline on both iCubWorld1.0 and CIFAR-10.
3. **tradition\_new\_cifar\_10.ipynb**: explores enhancements to the CV pipeline and further CIFAR-10 experiments.

Each notebook contains detailed explanations of the architecture, data preprocessing, hyperparameter tuning, and performance metrics.

## Key Results

* **CNN (Simple vs. Improved)**

  * Improved SimpleCNN with global average pooling achieved up to \~58% on the Categorization subset and \~88% on Demonstrator, outperforming the original flatten-based design.
  * Validation accuracy on iCubWorld1.0 reached \~98.8% with optimized hyperparameters.

* **Traditional CV (BoVW + SVM)**

  * Background subset accuracy: \~93–100% (background bias detected).
  * Demonstrator/Categorization/Robot subsets: 11–31% accuracy, highlighting limitations in generalization and spatial information.
  * CIFAR-10 pipeline achieved \~29% overall accuracy with a 500-word vocabulary and SVM.

For detailed tables, figures, and analysis, refer to the [project report](f77463ks_Kunwei_Song_Computer_Vision_Report.pdf).

Methods	Accuracy(%)
	Background	Categorization	Demonstrator	Robot
BoVW+SVM(tf–idf+L2)	93.0	31.08	21.59	11.47
BoVW+SVM(Stop-Words)	100.0	28.04	25.24	13.84
SimpleCNN（Flatten）	51.5	51.37	80.41	7.98
SimpleCNN（GAP）	53.0	58.02	88.51	16.46
ResNet-18	64.0	81.16	93.50	25.06
![image](https://github.com/user-attachments/assets/540050f6-6d77-474e-99a1-6c35ebca90c1)


## References

* LeCun et al., 1998. Gradient-based learning applied to document recognition.
* Krizhevsky et al., 2012. ImageNet classification with deep convolutional neural networks.
* Fanello et al., 2013. iCub World: Friendly Robots Help Building Good Vision Data-Sets.
* Lin et al., 2013. Network In Network.

(See `report.pdf` for the full reference list.)
