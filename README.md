# 📚 Kindle Review Sentiment Analysis: NLP Comparison

A comparative study of Sentiment Analysis on Kindle Book Reviews using three distinct vectorization techniques: **Bag of Words (BoW)**, **TF-IDF**, and **Word2Vec**.

---

## 🔄 Project Workflow

### 📊 Workflow Diagram (Mermaid)

```mermaid
flowchart TD
    A[Raw Data: all_kindle_review.csv] --> B[Preprocessing & Cleaning]

    B --> B1[Lowercasing]
    B --> B2[HTML Removal]
    B --> B3[Regex Cleaning]
    B --> B4[Stopwords Removal]
    B --> B5[Lemmatization]

    B --> C1[Bag of Words]
    B --> C2[TF-IDF]
    B --> C3[Word2Vec (Vector Sum)]

    C1 --> D1[Logistic Regression]
    C2 --> D2[Logistic Regression]
    C3 --> D3[Logistic Regression]

    D1 --> E[Model Comparison]
    D2 --> E
    D3 --> E

    E --> F[Streamlit Dashboard]
```

---

## 🚀 Project Overview

This project explores how different text-to-numerical conversion methods impact the accuracy of a sentiment classifier. It includes a complete pipeline from raw data preprocessing to deployment using a **Streamlit dashboard** for real-time predictions.

---

## 🧠 Core Results (Comparative Performance)

| Method       | Accuracy   | Feature Logic                 |
| ------------ | ---------- | ----------------------------- |
| **TF-IDF**   | **84.62%** | Weights words by importance   |
| **BoW**      | **84.00%** | Word frequency counts         |
| **Word2Vec** | **79.37%** | Semantic vector relationships |

---

## 🛠️ Technical Implementation

### 1️⃣ Data Preprocessing

* Removed HTML tags (BeautifulSoup)
* Removed URLs, special characters, and numbers
* Converted text to lowercase
* Removed stopwords
* Applied **WordNet Lemmatization**

---

### 2️⃣ Feature Engineering

#### 🔹 Bag of Words (BoW)

* Converts text into frequency-based vectors

#### 🔹 TF-IDF

* Highlights important words by reducing weight of common words

#### 🔹 Word2Vec (Vector Sum)

* 100-dimensional embeddings (RAM-optimized)
* Used **vector sum** instead of average to retain intensity of longer reviews

---

### 3️⃣ Classification

* **Logistic Regression**

  * Fast and efficient on CPU (i5)
  * Works well with both sparse and dense vectors

---

## 💻 System Constraints & Optimization

* **RAM Optimization**

  * Limited to 8GB → Used 100-dim Word2Vec
  * Avoided `.toarray()` on large sparse matrices

* **CPU Optimization**

  * Multi-core processing (`workers=4`) for Word2Vec training

---

## 📂 Project Structure

```text
├── all_kindle_review.csv     # Dataset
├── notebook.ipynb            # Training & experimentation
├── app.py                    # Streamlit app
├── requirements.txt          # Dependencies
├── *.pkl                     # Saved models & vectorizers
└── kindle_vectors.kv         # Word2Vec embeddings
```

---

## 🏃 How to Run Locally

### 1️⃣ Activate Environment

```bash
# Windows
.\kindle_venv\Scripts\activate
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the App

```bash
streamlit run app.py
```

---

## 📊 Conclusion

* **TF-IDF achieved the best performance (84.62%)**
* Indicates that **keyword importance matters more than semantic context** in this dataset
* Demonstrates an efficient NLP pipeline deployable on **standard consumer hardware**

---

## 🌟 Future Improvements

* Try Deep Learning models (LSTM / BERT)
* Hyperparameter tuning for Logistic Regression
* Use pretrained embeddings (GloVe, FastText)
* Deploy on cloud (AWS / Streamlit Cloud)

---

## 📬 Contact

Feel free to connect or contribute!
