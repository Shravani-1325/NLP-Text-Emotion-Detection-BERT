  # 📃🎭 Text Emotion Analysis using ML, LSTM & BERT
![Python](https://img.shields.io/badge/Python-3.12-blue)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-Project-green)
![Tensorflow](https://img.shields.io/badge/TensorFlow-DL-orange)
![Transformers](https://img.shields.io/badge/Transformers-BERT-yellow)
![NLP](https://img.shields.io/badge/NLP-Text%20Processing-green)

#### 🔗 Live Application: https://emotisense-text-emotion-detection.streamlit.app/
---

## 📖 1. Project Overview

Text Emotion Analysis is a Natural Language Processing (NLP) task focused on classifying human emotions from textual data. This project involves building a pipeline to categorize inputs like messages, tweets, or reviews into specific emotional states such as joy, sadness, anger, fear, love, and surprise.

Multiple models are implemented, including **Machine Learning**, **LSTM**, and **BERT**, and compared to find the best performance. The final model is deployed using Streamlit, and models are hosted on Hugging Face Hub.

---

## 📉 2. Problem Statement

In the modern digital era, individuals frequently share their feelings via text on social platforms, messaging apps, and review sites. However, computers often lack the innate ability to decode these human affects. This project addresses the need for automated systems that can interpret unstructured text to understand the underlying human sentiment.

---
## 🎯 3. Project Objectives

**Develop a predictive model** capable of analyzing unstructured text to identify underlying human sentiments.

> **Key Deliverables:**

***Accurately categorize emotions into specific classes*** -`Joy`, `Sadness`, `Anger`, `Fear`, `Love`, `Surprise` etc

***Compare multiple modeling approaches*** — from Naive Bayes and Logistic Regression to BiLSTM and BERT — to understand accuracy vs. complexity trade-offs.

---

## 📊 4. Data Understanding
This dataset is specifically curated for supervised emotion classification. Each text entry is mapped to a human psychological state, serving as the ground truth for model training.
>  **Dataset Source**
* **Source:** [Kaggle - Text Emotion](https://www.kaggle.com/datasets/prajwalnayakat/text-emotion)


> **Dataset Characteristics**
* ***Total Rows:***  ~93,000+ samples (after deduplication)
* ***Unique Emotion Classes:*** 11 (Neutral, Sadness, Happiness, Love, Surprise, Anger, Relief, Fun, Hate, Enthusiasm, Empty)


> **Feature Breakdown**

The file typically contains two primary columns:

| Column Name | Type | Description |
| :--- | :--- | :--- |
| **Text** | String | The input sentence, tweet, or message (e.g., *"I'm so thrilled about this!"*). |
| **Emotion** | Categorical | The target label representing the primary feeling (e.g., **joy**, **sadness**). |
---

## 🔍 5. Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to understand the data distribution, identify class imbalance, and detect issues like duplicates and noise before model building.

> #### Text Analysis & Visualizations
* ***Emotion Distribution:*** Visualized class frequencies; the dataset is largely balanced, with the "empty" class being the smallest (~3,500 samples).
* ***Text Statistics:*** Analyzed distribution of character counts, word counts, and sentence lengths to understand text structure and conversational patterns.
* ***Word Frequency*:** Identified dominant terms across the corpus and generated **WordClouds** per emotion to highlight class-specific keywords.
---

## 🧪 6.  Data Preprocessing (Data Cleaning)

Raw text data was standardized through a series of cleaning steps before feeding into NLP pipelines.

> #### Cleaning Steps

| Step | Description |
|---|---|
| **Lowercasing** | All text converted to lowercase to ensure uniformity (e.g., `"HAPPY"` → `"happy"`) |
| **Remove Punctuation** | All punctuation characters (`.,!?` etc.) stripped using `string.punctuation` |
| **Remove Numbers** | Digits removed using `char.isdigit()` filter |
| **Remove Special Characters** | Non-alphanumeric characters eliminated |
| **Remove Extra Spaces** | Multiple whitespace sequences collapsed into a single space using `" ".join(x.split())` |


---

## ⚙️ 7. NLP Preprocessing

Advanced text cleaning was performed using the nltk library to prepare data for vectorization:

 
| Step | Description | Example |
|---|---|---|
| **Tokenization** | Converted sentences into individual word tokens. | `"feeling happy"` → `["feeling", "happy"]` |
| **Stopword Removal** | Eliminated common, low-information words | `["i", "am", "so", "happy"]` → `["happy"]` |
| **Lemmatization** | Reduced words to their base or dictionary form using the `WordNetLemmatizer`. | `"feeling"` → `"feel"`, `"running"` → `"run"` |

## 📈 8. Feature Engineering
Converted processed text into numerical representations for the machine learning models:
>#### 8.1 TF-IDF Vectorization

**TF-IDF Vectorization** utilized `TfidfVectorizer` to transform text into numerical features, weighing terms based on their importance within the document and across the entire corpus.

> #### 8.2 Label Encoding

Categorical emotion labels were converted to integer codes so models can process them numerically.

---

## 🤖 9. Baseline ML Models

Three classical Machine Learning models were implemented on TF-IDF features to establish performance benchmarks before moving to deep learning approaches.

> #### Models :

* ***Naive Bayes:*** A probabilistic classifier chosen for its efficiency and strong performance in text classification tasks.
* ***Logistic Regression:*** A linear model used to establish a baseline for multiclass classification.
* ***Random Forest:*** An ensemble learning method used to capture non-linear relationships and improve robustness.

### 9.1 Model Evaluation Results
Each model was evaluated using a comprehensive suite of metrics such as Accuracy Precision, Recall & F1 Score to ensure a balanced assessment across all emotion classes:

| Model | Accuracy |
|---|---|
| 🥇 Random Forest | ~80.06% | 
| 🥈 Logistic Regression | ~80.02% | 
| 🥉 Naive Bayes | ~68.0% | 
---

## 🚀 10. Deep Learning (Sequence Models)

To capture the sequential nature and context of language, a Deep Learning approach was implemented:

> #### BiLSTM Model:

A Bidirectional Long Short-Term Memory (**BiLSTM Model**) was used to processes each sequence in both **forward** and **backward** directions, capturing context from both sides of a word — significantly improving emotion classification over a standard unidirectional LSTM.
 

### 10.1 Architectures Implemented

- ***Embedding Layer:*** Maps words into dense vectors of fixed size.
- ***Bidirectional LSTM Layer:*** Learns context from both past and future states in the sequence.
- ***Dense Layer:*** A fully connected layer with Softmax activation for multi-class emotion classification.

### 10.2 BiLSTM Model Evaluation

| Metric | Score |
|---|---|
| **Accuracy** | ~75.8% |
| **Weighted F1-Score** | ~76.20 |

Achieved ~75.8% accuracy with stable performance.

---

## 📊 11. Transformer Model (BERT)

A transformer-based model, BERT (Bidirectional Encoder Representations from Transformers), was utilized for advanced text understanding.

* ***Pre-trained Tokenizer:*** Used the `BertTokenizer` from Hugging Face to ensure the text is formatted exactly as the model was originally trained.
* ***Fine-tuning:*** The pre-trained `BERT-base-uncased` model was fine-tuned on the emotion dataset to adapt its deep linguistic knowledge to this specific classification task.
* ***Framework:*** Implemented using the **Hugging Face Transformers** library.

### 11.1 BERT Model Evaluation :

| Metric | Score |
|---|---|
| **Accuracy** | ~84% |
| **Weighted F1-Score** | ~85% |

>**Key Insights:**
- BERT achieved the **best overall performance** across all models, with ~84% accuracy and ~85% F1-score.
- Its bidirectional attention mechanism enables it to understand subtle emotional cues and contextual nuances that TF-IDF and LSTM models miss.

---

## 🏆 12. Model Comparisons

The models were benchmarked based on their ability to classify 11 distinct emotions. While traditional ML models provided a strong baseline, Deep Learning and Transformer models showed superior performance in capturing linguistic context.
 
### 📊 Results Table
 
| Model | Accuracy | F1-Score | Notes |
| :--- | :--- | :--- | :--- |
| 🥇**BERT (Transformer)** | **~84.00%** | **~85%** | **Best overall performance; utilizes deep bidirectional context.** |
| 🥈**Random Forest** | **~80.90%** | **~80%** | **Highest performing classical ML model; handles non-linear patterns.** |
| 🥉**Logistic Regression** | **~80.60%** | **~80%** | **Highly effective linear baseline using TF-IDF features.** |
| **BiLSTM** | **~75.80%** | **~75%** | **Captured sequence data but limited by fixed vocabulary size.** |
| **Naive Bayes** | **~68.43%** | **~68%** | **Fastest training time; serves as the initial project baseline.** |

---
## 📌 13. Best Model
`BERT` outperformed all other models with 84% accuracy and 85% F1-score, proving that pretrained transformer models generalize best for fine-grained emotion classification.

---
## ⚙️ 14. Error Analysis

* ***Neutral vs. Empty:*** The models occasionally confused "neutral" and "empty" due to similar linguistic structures in short-form text.
* ***Slight Imbalance:*** The "empty" class (~3,507 samples) had lower recall compared to "happiness" or "sadness," which were more prevalent.
* ***Contextual Nuance:*** Sarcasm and dual-emotion sentences (e.g., "I'm so happy I could cry") were the primary sources of misclassification for non-Transformer models.

---

## 📃 15. Hugging Face Hub & Deployment
* ***Hugging Face:*** The fine-tuned BERT model and tokenizer have been uploaded to the Hugging Face Model Hub for easy integration via the `transformers` library.
* ***Live App:*** The Streamlit application is deployed and accessible, providing a user-friendly interface for the emotion classification engine.

---

## 👩‍💻 14. Author

> **Shravani More**
Computer Science & Electronics Student 


