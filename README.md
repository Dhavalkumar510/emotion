# Emotion Detection in Text Using BERT

## 📚 Introduction
Emotion detection in text is a growing area in Natural Language Processing (NLP) that involves identifying emotional tones such as joy, fear, love, anger, sadness, and surprise. It has practical applications in sentiment analysis, customer feedback evaluation, and social media monitoring.

This project explores the use of **BERT-based large language models** for emotion classification. BERT (Bidirectional Encoder Representations from Transformers) enhances language understanding by analyzing the context of words in both directions. The model was trained and fine-tuned on the **Emotion Dataset** to detect emotions in text with high accuracy.

---

## 🛠️ Methodology

### 📦 Dataset
- **Source**: [Emotion Dataset via `datasets` library](https://huggingface.co/datasets/dair-ai/emotion)
- **Split**:  
  - Training: 16,000 samples  
  - Validation: 2,000 samples  
  - Test: 2,000 samples  

The dataset was converted into Pandas DataFrames for exploration and visualization. Class distribution was visualized to address imbalance during training.

### 🔧 Preprocessing
- Tokenized text using `BertTokenizer` from Hugging Face’s Transformers library (`bert-base-uncased`)
- Applied truncation and padding to ensure uniform input length
- Used a custom function with `dataset.map()` for efficient preprocessing

### 🤖 Model Training
- **Model**: `bert-base-uncased` with sequence classification head
- **Training Setup**:
  - Learning Rate: 2e-5
  - Batch Size: 16
  - Epochs: 2
  - Weight Decay: 0.01
- Used Hugging Face’s `Trainer` API for training and evaluation
- Metrics: Accuracy, F1 Score, Precision, Recall

---

## 📊 Results

| Epoch | Training Loss | Validation Loss | Accuracy | F1 Score | Precision | Recall |
|-------|----------------|------------------|----------|-----------|------------|--------|
| 1     | 0.2374         | 0.2101           | 92.60%   | 0.8861    | 0.8779     | 0.8996 |
| 2     | 0.1171         | 0.1751           | 92.45%   | 0.8794    | 0.8804     | 0.8785 |

The BERT-based model achieved **high accuracy and balanced performance** across all emotion categories.

---

## 🔍 Prediction & Evaluation
Post-training, the model was tested on various unseen texts and successfully predicted corresponding emotions. This highlights the model's robustness and generalization capabilities.

---

## ✅ Conclusion
The BERT-based model proved to be highly effective for text-based emotion classification. It achieved strong results with consistent metrics. Future work includes exploring other transformer architectures and experimenting with data augmentation techniques to further enhance performance.

---


## 📚 References

- Devlin, Jacob, et al. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* arXiv:1810.04805, 2018.
- [Hugging Face BERT Documentation](https://huggingface.co/docs/transformers/en/model_doc/bert)
- [bert-base-uncased Model on Hugging Face](https://huggingface.co/google-bert/bert-base-uncased)

---
