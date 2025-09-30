# fake-news-detection-using-ML
Building a machine learning model to classify news articles as either real or fake using textual data.
📰 Fake News Detection using Machine Learning
📝 Project Overview

With the rapid growth of online media, fake news has become a major challenge in spreading misinformation. This project uses machine learning techniques to classify whether a news article is real or fake based on its content.

The project demonstrates Natural Language Processing (NLP) techniques such as text cleaning, vectorization (TF-IDF/CountVectorizer), and classification models to detect fake news efficiently.

🎯 Objectives

🔍 Analyze the dataset to understand patterns in fake vs real news

🧹 Apply text preprocessing (tokenization, stopword removal, stemming/lemmatization)

📊 Transform text into numerical vectors (Bag of Words, TF-IDF)

🤖 Train ML models (Logistic Regression, Naive Bayes, Random Forest, etc.)

📉 Evaluate model performance using accuracy, precision, recall, F1-score

📊 Visualize insights (word frequencies, confusion matrix)

📂 Dataset

Source: Kaggle Fake News Dataset
 or similar

Columns:

id – Unique identifier

title – News article headline

author – Author name

text – News article content

label – Target (1 = Fake, 0 = Real)

⚙️ Workflow

Data Preprocessing

Handle missing values

Clean text (remove punctuation, lowercase, stopwords, stemming/lemmatization)

Exploratory Data Analysis (EDA)

Word frequency distributions

Word clouds for fake vs real news

Distribution of labels (balanced/unbalanced dataset)

Feature Engineering

Vectorization: CountVectorizer & TF-IDF

N-grams for better context

Model Building

Train-test split

Models: Logistic Regression, Naive Bayes, Random Forest, Support Vector Machine

Hyperparameter tuning

Model Evaluation

Accuracy

Precision, Recall, F1-score

Confusion Matrix

Visualization

Most common fake vs real words

Model comparison bar chart

Confusion matrix heatmap

📊 Results

✅ Achieved ~0.xx accuracy on test data

✅ Naive Bayes/Logistic Regression performed best for text classification

✅ Visualizations reveal strong word usage differences between fake and real news
<img width="759" height="504" alt="image" src="https://github.com/user-attachments/assets/d1df769c-303d-46b8-83b6-6ae8174c4f81" />


🛠️ Tech Stack

Python: Pandas, Numpy, Scikit-learn

NLP: NLTK / spaCy, Scikit-learn Vectorizers

Visualization: Matplotlib, Seaborn, WordCloud

Jupyter/Kaggle Notebooks

🚀 Outcomes

Developed a baseline ML model to detect fake news articles

Built an interpretable NLP pipeline from preprocessing to prediction

Produced clear visualizations to understand dataset characteristics

📌 Future Work

Integrate deep learning models (LSTMs, Transformers like BERT)

Deploy as a web app (Flask/Streamlit) for live fake news detection

Expand dataset to include multilingual sources
