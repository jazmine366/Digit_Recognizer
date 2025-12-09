# 3️⃣ Digit Recognizer

A convolutional neural network (CNN) built using **PyTorch**, trained on the **MNIST handwritten digit dataset** for the Kaggle Digit Recognizer competition.  
The model reaches **98.6% validation accuracy** and a **0.987 Kaggle Public Score**, with **4.14× faster training** using Apple’s MPS GPU backend.

---

## Features
- Custom CNN built from scratch using PyTorch  
- Utilizes **Apple Metal Performance Shaders** (MPS) for GPU acceleration  
- 4.14× faster training compared to CPU  
- 0.987 Kaggle Public Score  
  Complete pipeline:
  - Data preprocessing -> Train/validation split -> CNN architecture -> Training + validation -> Kaggle CSV submission generation
---

## Model Architecture
Input (1×28×28)
   ↓
Conv2d (1→32, 3×3)
   ↓
ReLU
   ↓
MaxPool (2×2)
   ↓
Conv2d (32→64, 3×3)
   ↓
ReLU
   ↓
MaxPool (2×2)
   ↓
Flatten (64×5×5 → 1600)
   ↓
Linear (1600 → 128)
   ↓
ReLU
   ↓
Linear (128 → 10)

---

## Building the Model 
<img src="https://github.com/user-attachments/assets/288bd39f-6327-483b-b4dd-58905d871f8e" width="180">

I start with a 28×28 grayscale image shaped (1,28,28).
Raw pixel values contain both positive and negative information. Applying ReLU removes negative values, keeping only the strongest signals. This allows the network to detect straight segments, bends, and curves in the digits.

RELU = max(0, x) //positive x or just 0 

Without ReLU, outputs behave like straight lines with no change in slope. With ReLU, the network produces “bends and kinks” that represent the curves and corners of digits. 
Negative inputs are set to zero while positive inputs pass through, highlighting meaningful features and enabling the CNN to learn the essential shapes in the digits for classification.

---

## CPU vs MPS GPU Training Speed
The same model was trained for **8 epochs** on CPU and on Apple’s MPS GPU.

| Device | Train Time |
|--------|-------------------------|
| **CPU** | 89.08 seconds |
| **MPS GPU** | 21.54 seconds |
| **Speedup** | **4.14× faster** |

---

## Results
- **Validation Accuracy:** 98.64%  
- **Best Kaggle Score:** 0.98728  
- **Best Epoch Count:** 8  



