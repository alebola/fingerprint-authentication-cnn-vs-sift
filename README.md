# Fingerprint Authentication: CNN vs SIFT

## 📌 Project Overview
This project explores **biometric fingerprint authentication** by testing two different hypotheses:

1. **Feature-based approach (SIFT)** – extracting handcrafted descriptors from fingerprint images.  
2. **Deep learning approach (CNN)** – training a Convolutional Neural Network for binary classification.  

The goal was not only to compare performance but also to design and implement two fundamentally different pipelines for the same task.


## 🚀 Key Highlights
- **Machine Learning & AI focus**: Applied both classical computer vision (SIFT) and deep learning (CNN) to the same biometric authentication problem.  
- **Experiment-driven workflow**: Developed, trained, and evaluated both approaches under a controlled setup.  
- **End-to-end pipeline**: From dataset preprocessing to training, evaluation, and visualization of metrics.  
- **Skills**:  
  - ✅ PyTorch (CNN model design, training loop, augmentation, early stopping, threshold tuning)  
  - ✅ OpenCV (SIFT feature extraction, matching, evaluation)  
  - ✅ Scikit-learn (metrics: ROC, EER, F1, accuracy)  
  - ✅ Data augmentation and preprocessing for small/low-quality datasets  


## 📂 Dataset
The dataset was **limited in size** and consisted of **low-quality fingerprint images**.  
This constraint was important to simulate a realistic challenge, but it also explains why the absolute results are not the main focus.  
The **core contribution** is the **design and validation of two different hypotheses** for fingerprint authentication.  


## 📊 Results
- **SIFT approach**: Achieved consistently strong results, with very low error rates (EER close to 0).  
- **CNN approach**: Performance was more unstable due to the small dataset, but showed the viability of deep learning with proper scaling.  
- **Conclusion**: In this setting, **SIFT outperformed CNN**.  


## 🛠️ Technologies Used
- **Python**
- **PyTorch** (deep learning, CNN design and training)
- **OpenCV** (SIFT feature extraction, preprocessing, augmentation)
- **Scikit-learn** (evaluation metrics and ROC curves)
- **Matplotlib** (visualizations)


## 📌 Takeaways
- Designed and validated **two hypotheses** for fingerprint authentication (SIFT and CNN).  
- Learned how dataset quality and size directly affect model performance.  
- Gained practical experience with both **classical computer vision** and **deep learning pipelines**.   


## ⚡ How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/alebola/fingerprint-authentication-cnn-vs-sift.git
   ```
2. Install dependencies:
   ```bash
   pip install torch torchvision scikit-learn opencv-python matplotlib pillow
    ```
3. Run the notebooks in order:
   - notebooks/00_preprocessing.ipynb → build the dataset structure
   - notebooks/01_sift.ipynb → test fingerprint authentication with SIFT
   - notebooks/02_cnn.ipynb → train and evaluate the CNN approach
