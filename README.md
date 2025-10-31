# Skin Disease Classification Using Deep Learning  

##  Overview  
This project aims to accurately classify various **skin diseases** from dermoscopic images using a **deep learning-based approach**.  
A **ResNet50** convolutional neural network (CNN) was employed as the feature extractor, combined with a **Cosine Annealing Learning Rate Scheduler with Warmup** to enhance model convergence and performance.  

The system supports the classification of **15 distinct skin conditions**, contributing to early and reliable disease detection that can assist medical professionals in diagnostic processes.

---

## Dataset Summary  
The dataset used consists of dermoscopic images organized into 15 classes:

- Acne  
- Actinic Keratosis  
- Basal Cell Carcinoma  
- Chickenpox  
- Dermato Fibroma  
- Dyshidrotic Eczema  
- Melanoma  
- Nail Fungus  
- Nevus  
- Normal Skin  
- Pigmented Benign Keratosis  
- Ringworm  
- Seborrheic Keratosis  
- Squamous Cell Carcinoma  
- Vascular Lesion 

- **Training Images:** 33,720  
- **Validation Images:** 7,215  
- **Testing Images:** 7,215  

---

##  Model Architecture  
The model is based on **ResNet50** pretrained on **ImageNet**, with the top layers replaced by custom fully connected layers for skin disease classification.

**Architecture Summary:**
1. **Base Model:** ResNet50 (frozen convolutional layers)  
2. **GlobalAveragePooling2D** layer for feature aggregation  
3. **Fully Connected Dense Layer (512 units)** with ReLU activation  
4. **Dropout Layers (0.4 and 0.3)** to reduce overfitting  
5. **Softmax Output Layer** with 15 neurons for multi-class prediction  

**Optimizer:** Adam  
**Loss Function:** Categorical Crossentropy  
**Learning Rate:** Cosine Annealing with Warmup  
**Batch Size:** 32  
**Input Size:** 224 × 224 × 3  

---

##  Training Strategy  

### Data Augmentation
To enhance generalization, the following augmentations were applied:
- Random rotations (±20°)
- Width and height shifts (±10%)
- Shear and zoom transformations
- Horizontal flips  

### Learning Rate Scheduling  
A **Cosine Annealing Learning Rate Scheduler with Warmup** was implemented:  
- **Initial LR:** 1e-4  
- **Warmup Epochs:** 2  
- **Total Epochs:** 10  

This strategy enables a smooth learning rate transition, improving stability and convergence speed.

---

##  Training Results  

| Metric | Validation Set | Test Set |
|---------|----------------|-----------|
| **Accuracy** | 86.57% | 86.61% |
| **Loss** | 0.3826 | 0.3707 |

**Training Log Highlights:**
- The validation accuracy steadily improved from **76.6% (Epoch 1)** to **86.6% (Epoch 10)**.  
- EarlyStopping and ModelCheckpoint ensured optimal model performance retention.  

**Training completed successfully!**  
Model saved as: `best_resnet50_scheduler_10epochs.h5`

--

---

##  Technologies Used  
- **Language:** Python  
- **Frameworks:** TensorFlow, Keras  
- **Libraries:** NumPy, Pandas, Matplotlib, Seaborn  
- **Pretrained Model:** ResNet50 (ImageNet weights)  
- **Scheduler:** Cosine Annealing with Warmup  

---

## 📈 Future Work  
- Fine-tune deeper layers of ResNet50 for improved accuracy  
- Incorporate ensemble CNNs or Vision Transformers (ViT)  


## 👨‍💻 Author  
**Banoth Ajay Vardhan Naik**  
🎓 *IIT Bhubaneswar | Dual Degree (B.Tech + M.Tech) in Computer Science*  
📧 *Email:* ajayvardhannaik@gmail.com
