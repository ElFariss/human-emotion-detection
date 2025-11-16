# Human Emotion Detection  
A project focused on comparing **Deep Neural Network (Vision Transformer)** and a **Hybrid Model combining Polynomial Logistic Regression + Attention Layer** for human facial emotion classification.

The goal is to evaluate whether classical statistical modeling (polynomial logistic regression) boosted with an attention mechanism can outperform a pure DNN (ViT) on a large-scale facial emotion dataset.

---

## Overview  
This repository contains experiments and notebooks for detecting human emotions from **single human faces**.  
There are **two main modeling approaches**:

1. **Vision Transformer (DNN-based)**
2. **Hybrid Model: Polynomial Logistic Regression + Attention Layer**

A third folder (HOG + SVM) exists but is currently an unfinished attempt to perform image segmentation for detecting multiple faces simultaneously.

---

## Models Compared

### Vision Transformer (ViT)  
This pure DNN model serves as the baseline. While ViT typically performs strongly in image classification, on this dataset the model shows signs of class imbalance and difficulty in differentiating subtle expressions such as *Fear* and *Sad*.

#### **Classification Report**

```

          precision    recall  f1-score   support

   Angry     0.3327    0.4781    0.3924     10148
    Fear     0.3459    0.1220    0.1803      9732
   Happy     0.7354    0.6728    0.7027     18439
     Sad     0.4744    0.3083    0.3737     12553
 Suprise     0.4367    0.8524    0.5776      8227

accuracy                         0.4963     59099

macro avg 0.4650 0.4867 0.4453 59099
weighted avg 0.5051 0.4963 0.4761 59099
```
#### **Confusion Matrix**
```
[[ 4852 614 1419 1390 1873]
[ 2434 1187 1052 1566 3493]
[ 2501 380 12406 1157 1995]
[ 4371 974 1654 3870 1684]
[ 425 277 338 174 7013]]

```

**Key Insight:**  
- ViT performs very well on **Happy** and **Surprise**,  
- Struggles heavily on **Fear** and **Sad**,  
- Overall accuracy: **49.63%**  
This makes ViT a reasonable baseline but not optimal for emotion recognition, where subtle differences matter.

---

### Hybrid Model: Polynomial Logistic Regression + Attention  
This model applies:

- Polynomial feature expansion (captures more non-linear boundaries than linear logistic regression)  
- An **attention mechanism** to reweight important facial features  
- Lightweight architecture compared to ViT  

Despite being simpler than a full transformer, this hybrid approach produced **significantly higher accuracy and macro-F1**.

#### **Hybrid Model Performance**
```
Train Loss: 0.1734 | Val Loss: 0.4597
Val Accuracy: 87.27% | Macro-F1: 0.8659
Saved best model (F1=0.8659)
```

**Key Insight:**  
- The hybrid model generalizes far better than ViT  
- Attention helps focus on emotion-relevant regions (eyes, mouth, eyebrows)  
- Polynomial logistic regression remains interpretable while benefiting from feature expansion  
- **Huge improvement** over the ViT baseline (49% → 87%)

---

## Notebooks  
| Notebook | Description |
|----------|-------------|
| `EDA.ipynb` | Basic dataset exploration and visualization. |
| `human_face_emotion_ViT.ipynb` | Vision Transformer baseline training + evaluation. |
| `human_face_hybrid.ipynb` | Hybrid model training using polynomial logistic regression + attention. |
| `predict_hybrid.ipynb` | Inference notebook for the hybrid model. |
| `human_face_emotion_hogsvm.ipynb` | (Unfinished) Attempt at segmentation + classical HOG features. |

---

## HOG + SVM Status  
This approach was meant to segment images and detect **multiple faces per frame**, but it is **not yet complete**.  
It is included only as an experiment and not part of the main comparison.

---

## Conclusion  
The project demonstrates that:

- **A well-structured hybrid model can outperform a heavy DNN like ViT** for emotion classification.  
- Classical statistical models, when paired with attention mechanisms, can deliver *state-of-the-art performance* while remaining efficient and interpretable.  
- ViT baseline provides a useful comparison to highlight how emotion detection requires more focused, localized features than general image classification.

---

## Author  
Created by **Faris (ElFariss)**.  

---
