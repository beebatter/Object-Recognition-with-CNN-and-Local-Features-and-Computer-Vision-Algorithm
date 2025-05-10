# **Object Recognition Project (Computer Vision Assignment)**

**Author**: Kunwei Song

**Student ID**: 11537231

## **1. Project Overview**

This project investigates **object recognition** by comparing **lightweight convolutional neural networks (CNNs)** with a **traditional computer-vision pipeline** based on Bag-of-Visual-Words and SVM. Experiments are conducted on the **iCubWorld1.0** and **CIFAR-10** datasets to evaluate end-to-end learning versus handcrafted feature approaches.

## **2. Dataset**

We use two publicly available datasets:

* **iCubWorld1.0**: A robotics vision dataset with images of 10 object categories captured under various conditions.

  * **Training Samples**: 600 images (3 object instances × 200 images)
  * **Test Subsets**: 4 partitions (Background, Categorization, Demonstrator, Robot)
* **CIFAR-10**: A standard benchmark for image classification containing 60,000 32×32 color images in 10 classes.

  * **Training Samples**: 50,000
  * **Test Samples**: 10,000

These datasets provide complementary challenges: iCubWorld1.0 tests robustness to background and viewpoint changes, while CIFAR-10 measures performance on small-scale natural images.

## **3. Methods Implemented**

We implement and evaluate two main object-recognition pipelines:

### **Method 1: Bag-of-Visual-Words + SVM**

* **Feature Extraction**: SIFT descriptors on densely sampled keypoints.
* **Vocabulary Construction**: K-means clustering to form a 500-word codebook.
* **Encoding**: TF–IDF weighting with L2 normalization and stop-word removal variants.
* **Classification**: Linear SVM on histogram representations.

### **Method 2: Convolutional Neural Networks (CNNs)**

* **SimpleCNN (Flatten)**: A small CNN with fully connected layers after convolution.
* **SimpleCNN (GAP)**: Same architecture but replaces flatten with Global Average Pooling.
* **ResNet‑18**: A pre‑activated residual network fine‑tuned on each dataset.

## **4. Installation and Dependencies**


Typical dependencies include:

* numpy
* opencv-python
* scikit-learn
* matplotlib
* tqdm
* torch
* torchvision
* jupyter


Clone this repository and install required packages:

```bash
git clone <your-repo-url>  
cd <repo-directory>  
pip install -r requirements.txt  
```

### **Notebooks**

| Notebook                                  | Description                                                            |
| ----------------------------------------- | ---------------------------------------------------------------------- |
| `iCubWorld1_CNN_Improved.ipynb`           | Trains and evaluates SimpleCNN and ResNet‑18 on iCubWorld1.0.          |
| `iCubWorld_Traditional_CV_Improved.ipynb` | Implements BoVW + SVM pipeline on iCubWorld1.0 with improved settings. |
| `Cifar_10_Tradition_CV.ipynb`             | Applies BoVW + SVM on CIFAR-10 and compares with CNN results.          |

### **Requirements**

```text
numpy>=1.18.5  
opencv-python>=4.5.0  
matplotlib>=3.2.2  
scikit-learn>=0.24.0  
torch>=1.7.0  
torchvision>=0.8.1  
tqdm>=4.50.0  
jupyter>=1.0.0  
```

## **5. Running the Code**

1. Ensure **iCubWorld1.0** data is in `Dataset/iCubWorld1/train` and `Dataset/iCubWorld1/test`.
2. Launch Jupyter Lab or Notebook:

```bash

jupyter notebook

```
3. Open and run notebooks in order:  
   - `iCubWorld1_CNN_Improved.ipynb`  
   - `iCubWorld_Traditional_CV_Improved.ipynb`  
   - `Cifar_10_Tradition_CV.ipynb`  
4. Each notebook handles preprocessing, training, evaluation, and visualization of results.

## **6. Evaluation**  
The following table summarizes classification accuracy (%) on the iCubWorld1.0 test subsets:  

| Method                       | Background | Categorization | Demonstrator | Robot  |  
|------------------------------|------------|----------------|--------------|--------|  
| BoVW+SVM (tf–idf+L2)         | 93.0       | 31.08          | 21.59        | 11.47  |  
| BoVW+SVM (Stop-Words)        | 100.0      | 28.04          | 25.24        | 13.84  |  
| SimpleCNN (Flatten)          | 51.5       | 51.37          | 80.41        | 7.98   |  
| SimpleCNN (GAP)              | 53.0       | 58.02          | 88.51        | 16.46  |  
| ResNet-18                    | 64.0       | 81.16          | 93.50        | 25.06  |  

Accuracy on CIFAR-10 is also reported in the notebooks for direct comparison.

## **7. Results and Discussion**  
- **CNNs outperform** traditional CV pipelines on challenging subsets (Categorization, Demonstrator) due to learned spatial features.  
- **BoVW+SVM** excels on Background bias but suffers on generalization (Robot subset).  
- **Global Average Pooling** improves SimpleCNN stability and performance over flatten-based architecture.  
- **ResNet‑18** achieves closest performance to human baselines, demonstrating the benefit of residual learning.

## **8. Use of Generative AI Tools**  
- **ChatGPT** was used for code troubleshooting, documentation drafting, and conceptual discussion.  
- All AI-suggested code was manually reviewed and validated before inclusion.

## **9. Author**  
- **Kunwei Song** (Student ID: 11537231)

## **10. References**  
- Fanello et al., *iCubWorld: Friendly Robots Help Building Good Vision Data-Sets*, IROS 2013.  
- Krizhevsky et al., *ImageNet Classification with Deep Convolutional Neural Networks*, NIPS 2012.  
- Lin et al., *Network In Network*, ICLR 2014.  
- LeCun et al., *Gradient-based Learning Applied to Document Recognition*, IEEE 1998.  
- CIFAR-10 Dataset: [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)  
- iCubWorld1.0 Dataset: [http://robots.epfl.ch/icubworld1](http://robots.epfl.ch/icubworld1)

```
