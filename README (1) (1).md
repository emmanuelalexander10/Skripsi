# Perbandingan Metode Machine Learning dan Deep Learning pada Analisis Sentimen Review Produk E-Commerce Menggunakan Naïve Bayes, SVM, dan IndoBERT

Undergraduate thesis (Skripsi) implementation comparing conventional Machine Learning and Transformer-based Deep Learning methods for Indonesian-language sentiment analysis on e-commerce product reviews.

📄 **Paper**: Skripsi, Program Studi Sistem Informasi, Universitas Multimedia Nusantara (2026)

---

## 📋 Overview

Online product reviews are one of the main sources consumers use to judge product quality and satisfaction, but e-commerce sellers often receive far more reviews than can be read manually. This project builds and compares three sentiment classification approaches on Indonesian-language Tokopedia product reviews:

- **Naïve Bayes** and **Support Vector Machine (SVM)** — conventional Machine Learning models using TF-IDF feature representations
- **IndoBERT** — a transformer-based Deep Learning model fine-tuned for Indonesian text

The research follows the **CRISP-DM** framework (Business Understanding → Data Understanding → Data Preparation → Modeling → Evaluation → Deployment) and classifies each review into one of three sentiment classes: **Positive**, **Neutral**, and **Negative**, derived from the review's star rating.

The best-performing model (IndoBERT) is deployed as an interactive **Streamlit** web app that predicts the sentiment of a review in real time, along with a confidence score.

### Key Features

- **Three-way model comparison**: Naïve Bayes, SVM, and IndoBERT evaluated side by side in one consistent experimental pipeline, rather than in isolation as in most prior work
- **Real-world scraped dataset**: 15,000 Indonesian-language product reviews collected directly from Tokopedia (not a secondary/downloaded dataset)
- **Full CRISP-DM pipeline**: text normalization, tokenization, stemming, class balancing, model training, and evaluation
- **Class imbalance handling**: combination of Random Under Sampling and SMOTE to balance the minority (negative/neutral) classes
- **Enriched feature engineering for SVM**: word-level + character n-gram TF-IDF to better capture morphological variation (e.g. "kecewa" vs "mengecewakan")
- **Deployed prototype**: Streamlit web app with live sentiment prediction and confidence score

---

## 🎯 Key Contributions

1. **Simultaneous comparative benchmark**: unlike prior studies that compare methods in isolated pairs, this study benchmarks Naïve Bayes, SVM, and IndoBERT together under the same dataset, preprocessing, and evaluation protocol
2. **Primary, self-collected dataset**: 15,000 cleaned Indonesian reviews scraped directly from Tokopedia across multiple product categories (electronics, fashion, sports, mobile phones, tools), spanning review dates from 2019–2025
3. **Practical deployment**: results are translated into a working Streamlit-based sentiment analysis prototype usable by sellers, customer service teams, or researchers
4. **Diagnostic analysis of the neutral class**: the study shows that the persistently low performance on neutral reviews across all three models is a linguistic/data characteristic (ambiguous mixed-polarity text), not a modeling deficiency — motivating future aspect-based sentiment analysis (ABSA)

---

## Project Architecture / Research Workflow

This project follows the **CRISP-DM** methodology:
<img width="1267" height="490" alt="Architecture-Workflow" src="https://github.com/user-attachments/assets/cc264661-40b1-4fd7-a314-d861f0f53bc9" />
```

1. Business Understanding
   → Define the problem: sellers cannot manually process thousands of reviews

2. Data Understanding
   → Explore raw scraped data (text, rating, product_name, category)

3. Data Preparation
   → Data cleaning (remove duplicates, empty reviews, non-Indonesian/invalid text)
   → Text Normalization → Tokenization → Stemming
   → Label reviews as Positive / Neutral / Negative based on star rating

4. Modeling
   → Data Splitting (80:20 train/test)
   → Class balancing (Random Under Sampling + SMOTE)
   → Train Naïve Bayes, SVM (TF-IDF), and IndoBERT (fine-tuned transformer)

5. Evaluation
   → Classification report (Precision, Recall, F1-Score, Accuracy)
   → Confusion matrix per model

6. Deployment
   → Streamlit web app using the best-performing model (IndoBERT)
```

---

## 🚀 Installation

### Requirements

- Python 3.9+
- pandas, numpy, scikit-learn
- imbalanced-learn (for SMOTE / Random Under Sampling)
- Hugging Face `transformers` + PyTorch (for IndoBERT fine-tuning)
- Streamlit (for the deployed web app)
- Selenium (for the data scraping script)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd <project-folder>
```

2. Create a virtual environment:
```bash
conda create -n sentiment-analysis python=3.9
conda activate sentiment-analysis
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the Streamlit app:
```bash
streamlit run app.py
```

---

## 📊 Dataset Preparation

### Tokopedia Product Review Dataset

This is a **primary dataset**, collected directly via web scraping (not downloaded from an existing repository).

- **Source**: Tokopedia product review pages
- **Collection method**: Web scraping using Selenium (Python), conducted over ~5 hours in May 2025
- **Review period covered**: 2019–2025 (to capture varied market conditions and consumer behavior over time)
- **Raw scraped reviews**: 18,742
- **After data cleaning** (removing duplicates, empty reviews, non-Indonesian/invalid entries): **15,000 reviews**
- **Categories covered**: electronics, fashion, sports, mobile phones, tools, among others

**Attributes:**

| Column | Description |
|---|---|
| `text` | Unstructured review text (main input for sentiment analysis) |
| `rating` | Numeric star rating (1–5), used to derive the sentiment label |
| `product_name` | Name of the reviewed product |
| `category` | Product category |

**Train/test split**: 80:20 → 12,000 training reviews / 3,000 test reviews. The training set is then balanced using Random Under Sampling + SMOTE so each of the three sentiment classes is equally represented.

### Download

The dataset is not publicly redistributed due to it being scraped directly from Tokopedia for this research.

```
project/
├── data/
│   ├── raw/                # raw scraped reviews (18,742 rows)
│   └── processed/          # cleaned, labeled dataset (15,000 rows)
```

---

## 🏋️ Training

Three models are trained and compared:

**1. Naïve Bayes** — TF-IDF features, trained on the class-balanced training set.

**2. Support Vector Machine (SVM)** — linear kernel, `class_weight="balanced"`, using combined word-level + character n-gram TF-IDF features.

**3. IndoBERT** — fine-tuned using Hugging Face `transformers`:

| Parameter | Setting |
|---|---|
| Pretrained model | `indobenchmark/indobert-base-p1` |
| Learning rate | 2e-5 |
| Batch size (train) | 8 |
| Epochs | 3 |
| Max sequence length | 128 tokens |
| Optimizer | AdamW |
| Evaluation strategy | per epoch |
| Logging steps | 50 |

Example execution:
```bash
python train_naive_bayes.py
python train_svm.py
python train_indobert.py --config configs/indobert.yaml
```

> IndoBERT fine-tuning took ~8 minutes with batch size 8, limited by available GPU memory. Naïve Bayes and SVM train in seconds to a few minutes on CPU alone.

---

## 📊 Results

Final evaluation on the 3,000-review test set (classification report, macro-averaged):

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
|---|---|---|---|---|
| Naïve Bayes | 74.23% | 0.41 | 0.61 | 0.41 |
| SVM | 78.83% | 0.47 | 0.55 | 0.49 |
| **IndoBERT** | **94%** | **0.65** | **0.57** | **0.59** |

**Key findings:**
- IndoBERT consistently outperformed both conventional ML methods, with the largest gains on the minority **negative** class (F1-Score jumped from ~0.20–0.38 in NB/SVM to 0.58 in IndoBERT)
- All three models struggled with the **neutral** class (F1-Score ≤ 0.22 even for IndoBERT), indicating this is an inherent linguistic ambiguity issue (mixed-polarity reviews) rather than a modeling weakness
- IndoBERT's advantage comes from contextual embeddings, bidirectional attention (better handling of negation, e.g. "tidak mengecewakan"), and Indonesian-language pretraining — at the cost of higher compute requirements (GPU, longer training/inference time) compared to TF-IDF-based methods

The best model (IndoBERT) was deployed as a Streamlit app that classifies a pasted review as Positive / Neutral / Negative with a confidence score (e.g. "Positive (97.4%)").

---

## 🏗️ Project Structure

```
project/
├── scraping/                  # Selenium-based Tokopedia review scraper
├── preprocessing/             # Text normalization, tokenization, stemming, labeling
├── models/
│   ├── naive_bayes.py         # Naïve Bayes training & evaluation
│   ├── svm.py                 # SVM training & evaluation (TF-IDF word + char n-gram)
│   └── indobert/              # IndoBERT fine-tuning scripts
├── app.py                     # Streamlit deployment app
├── data/                      # Raw and processed datasets
└── requirements.txt           # Python dependencies
```

---

## 🙏 Acknowledgments

- Big Data Lab, Information Systems Study Program, Universitas Multimedia Nusantara (UMN)
- Supervisor: Suryasari, S.Kom., M.T.

---

## 📧 Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact: emmanuel.bloem@student.umn.ac.id

---

## 📜 License

This project is released under the MIT License. See [LICENSE](LICENSE) file for details.
