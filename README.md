# Book Recommender System
A Book Recommender System - Using both Content-Based and Collaborative Filtering approaches
Built using Python with libraries such as sklearn, pyspark, pandas, numpy etc. + GUI using tkinter

## Features:

### Content-Based Filtering:
Recommendations based on book metadata such as descriptions, genres, and author information.

Utilized a variety of NLP Techniques:
  - Text pre-processing: tokenization, stop-word removal, special character removal, stemming/lemmatization.
  - Vectorization:
      - Count Vectorizer for simple frequency-based features.
      - Tf-Idf (Term Frequency - Inverse Document Frequency) for weighted textual representations.
      - Word2Vec for capturing semantic relationships between words.

### Collaborative Filtering
Recommendations based on user preferences and ratings.
- Implemented using ALS (Alternating Least Squares) algorithm via PySpark for matrix factorization.

## Dataset used: 

A merge between Abdallah Wagih Ibrahim’s [Books Dataset](https://www.kaggle.com/datasets/abdallahwagih/books-dataset/data). and Zygmunt Zajac [Goodreads 10k](https://github.com/zygmuntz/goodbooks-10k).
After pre-processing and merging, we obtained over 2.4k book entries with metadata about the books and user reviews, used for the recommendation process

## GUI

**Collaborative Filtering Recommendations**: Get personalized recommendations based on past ratings.

<img src="screenshots/collab_filtering_GUI.png" width="640" height="370" />

**Content-Based Recommendations**: Choose a book you like and get similar suggestions.

<img src="screenshots/content_based_GUI.png?" width="640" height="370" /> 

## Pre-processing & filtering

The data went thorough pre-processing to improve recommendation quality by:
- Merging multiple editions of the same book into single entries.
- Removing books with missing or incomplete metadata (e.g., no descriptions or genres).
-  Text cleaning: stop-word removal, removal of special characters and punctuation, lowercasing and tokenization
- Converting textual data into numerical features using advanced NLP techniques.

For Collaborative Filtering, the dataset was transformed into a user-item interaction matrix and normalized for training the ALS model.

---

Feel free to explore the repository, test the models, and customize the recommendations!

