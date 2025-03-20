# 🧠 Sarcasm Detection in Tamil & Malayalam – Data Preprocessing  

### 🔍 Project Overview  
This repository is part of a larger project aimed at **detecting sarcasm** in Tamil and Malayalam text using deep learning models. My contribution focuses on **data preprocessing**, which is crucial for improving model performance.  

### ✨ My Contribution  
I worked on cleaning and preprocessing the dataset to ensure high-quality input for the model. My tasks included:  
✔ **Text Cleaning** – Removing special characters, URLs, and unnecessary symbols  
✔ **Tokenization** – Splitting text into meaningful words  
✔ **Stopword Removal** – Eliminating common but unimportant words  
✔ **Normalization** – Handling variations in spelling and script differences  
✔ **Word Embeddings Preparation** – Converting text into numerical vectors for model input  

### 🛠️ Preprocessing Steps  
```python
import re
import nltk
from nltk.tokenize import word_tokenize

# Load stopwords for Tamil & Malayalam
stopwords_ta = set(open("stopwords_tamil.txt").read().split())
stopwords_ml = set(open("stopwords_malayalam.txt").read().split())

def clean_text(text):
    text = text.lower()  
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z\u0B80-\u0BFF\u0D00-\u0D7F\s]", "", text)  # Keep Tamil & Malayalam characters
    tokens = word_tokenize(text)  
    tokens = [word for word in tokens if word not in stopwords_ta and word not in stopwords_ml]  
    return " ".join(tokens)

# Example Usage
sample_text = "இந்த ஒரு உதாரணம்! 😂 http://example.com"
print(clean_text(sample_text))
```

### 📊 Impact of Preprocessing  
- **Reduced Noise**: Improved dataset quality by removing irrelevant content  
- **Better Model Training**: Enhanced word embeddings for sarcasm detection  
- **Efficient Tokenization**: Optimized text structure for deep learning models  

### 📌 Acknowledgment  
This contribution is part of a larger **Sarcasm Detection in Tamil & Malayalam** project.  

👉 *Excited to see how this dataset helps improve sarcasm detection!* 🚀
