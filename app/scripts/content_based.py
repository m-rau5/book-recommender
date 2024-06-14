from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import gensim as gs
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import warnings
import ast

warnings.filterwarnings("ignore", category=RuntimeWarning)


class Content_Recommender():
    def __init__(self, book_df_path, base_model):
        self.book_df = pd.read_csv(book_df_path)
        self.w2v = self.load_word2vec_model(base_model)

    def __convert_df_to_list(self, df, tokenized):
        if tokenized:
            header = ["Book Title", "Category", "Description (tokenized)"]
        else:
            header = ["Book Title", "Category", "Description"]
        list_of_lists = df.values.tolist()
        list_of_lists.insert(0, header)
        list_cropped = []

        for elem in list_of_lists:
            if len(elem[2]) > 300:
                elem[2] = elem[2][:300] + "..."
            list_cropped.append(elem)

        return list_cropped

    def TF_IDF(self, book_title, amount=10):
        temp_book_df = self.book_df.copy()

        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(temp_book_df['description'])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        if book_title in temp_book_df['Book-Title'].values:
            idx = temp_book_df.loc[temp_book_df['Book-Title']
                                   == book_title, 'Book-Id'].values[0]
        else:
            return None

        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:amount+1]
        book_indices = [i[0] for i in sim_scores]
        recommended_books = temp_book_df[[
            'Book-Title', 'categories', 'description']].iloc[book_indices]
        recommended_books_df = pd.DataFrame(recommended_books)

        return self.__convert_df_to_list(recommended_books_df, False)

    def Count_Vectorizer(self, book_title, amount=10):
        temp_book_df = self.book_df.copy()

        count_vectorizer = CountVectorizer(stop_words='english')
        count_matrix = count_vectorizer.fit_transform(
            temp_book_df['description'])
        cosine_sim = cosine_similarity(count_matrix, count_matrix)

        # Recommender:
        idx = temp_book_df[temp_book_df['Book-Title'] == book_title].index[0]

        # Sim scores
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:amount+1]

        book_indices = [i[0] for i in sim_scores]
        recommended_books = temp_book_df[[
            'Book-Title', 'categories', 'description']].iloc[book_indices]
        recommended_books_df = pd.DataFrame(recommended_books)
        return self.__convert_df_to_list(recommended_books_df, False)

    def Word_2_Vec(self, book_title, amount=10):

        temp_book_df = self.book_df.copy()

        # Get the description tokens of the target book
        description_tokens = temp_book_df.loc[temp_book_df['Book-Title']
                                              == book_title, 'tokenized_description'].iloc[0]

        # Used ast.literal_eval to transform the str of tokens into a python list
        description_tokens = ast.literal_eval(description_tokens)

        # comppute the avg vector of the TARGET book's description tokens
        target_vector = np.mean(
            [self.w2v.wv[token] for token in description_tokens if token in self.w2v.wv], axis=0).reshape(1, -1)

        similarity_scores = {}
        for index, row in temp_book_df.iterrows():
            if row['Book-Title'] != book_title:
                # compute the avg vector of the CURRENT book's description tokens
                book_vector = np.mean(
                    [self.w2v.wv[token] for token in ast.literal_eval(row['tokenized_description']) if token in self.w2v.wv], axis=0).reshape(1, -1)

                # we want to skip nan values
                if np.isnan(target_vector).any() or np.isnan(book_vector).any():
                    continue
                # get the cosine similarity
                similarity_scores[index] = cosine_similarity(
                    target_vector, book_vector)[0, 0]

        # sort the books by similarity score in descending order
        sorted_indices = sorted(
            similarity_scores, key=similarity_scores.get, reverse=True)
        # Return the most similar books

        # Only extract the needed columns
        recommended_df = temp_book_df.iloc[sorted_indices[:amount]]
        recommended_df = recommended_df[[
            "Book-Title", "categories", "description"]]

        return self.__convert_df_to_list(recommended_df, True)

    def load_word2vec_model(self, base_model):
        return gs.models.Word2Vec.load(base_model)
