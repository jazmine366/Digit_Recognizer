# Digit Recognizer

A convolutional neural network (CNN) built using **PyTorch**, trained on the **MNIST handwritten digit dataset** for the Kaggle Digit Recognizer competition.  
The model reaches **98.6% validation accuracy** and a **0.987 Kaggle Public Score**, with **4.14× faster training** using Apple’s MPS GPU backend.

---

## Features
- Custom CNN built from scratch using PyTorch  
- Utilizes **Apple Metal Performance Shaders (MPS)** for GPU acceleration  
- **4.14× faster training** compared to CPU  
- **0.987 Kaggle Public Score**  
- Complete pipeline:
  - Data preprocessing  
  - Train/validation split  
  - CNN architecture  
  - Training + validation  
  - Kaggle CSV submission generation  

---

## Model Architecture
- Conv2d(1 → 32, kernel=3)
- ReLU
- MaxPool2d(2)
- Conv2d(32 → 64, kernel=3)
- ReLU
- MaxPool2d(2)
- Flatten (64×5×5 → 1600)
- Linear 1600 → 128
- ReLU
- Linear 128 → 10

---

## CPU vs MPS GPU Training Speed

The same model was trained for **8 epochs** on CPU and on Apple’s MPS GPU.

| Device | Train Time (8 epochs) |
|--------|-------------------------|
| **CPU** | 89.08 seconds |
| **MPS GPU** | 21.54 seconds |
| **Speedup** | **4.14× faster** |

---

## Results
- **Validation Accuracy:** 98.64%  
- **Best Kaggle Score:** 0.98728  
- **Best Epoch Count:** 8  



