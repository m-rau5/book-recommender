from pyspark.sql import SparkSession
from pyspark.ml.tuning import CrossValidatorModel
from pyspark.sql.functions import col, explode,  expr, format_number
import os
import sys


# Env setup for spark to find python
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable


class Collab_Recommender():
    def __init__(self, book_df_path, ratings_df_path, base_model):
        # Setup Spark
        self.spark = SparkSession.builder \
            .master('local') \
            .appName('BRS-pyspark') \
            .config("spark.driver.memory", "12g") \
            .config("spark.driver.extraJavaOptions", "-Xss12M") \
            .getOrCreate()

        self.spark.conf.set(
            "spark.sql.execution.arrow.pyspark.enabled", "true")

        # Read the datasets
        self.books_df = self.spark.read.csv(
            book_df_path, header=True, inferSchema=True)

        self.book_ratings = self.spark.read.csv(
            ratings_df_path, header=True, inferSchema=True)

        # Initiate the initial Model
        self.load_model(base_model)

    def load_model(self, mPath):
        # Set the model we wish to use
        persistedModel = CrossValidatorModel.load(mPath)
        self.model = persistedModel.bestModel

    def get_recommendations(self, user_id):
        # Get the recommendations from the model we set
        user_recs = self.model.recommendForUserSubset(
            self.spark.createDataFrame([[user_id]], ["User-ID"]), numItems=10)

        # Use "explode" to separate rows on the recommendations array
        user_recs_exploded = user_recs.select(
            col("User-ID"), explode("recommendations").alias("recommendation"))

        # Get books and ratings + join on their name
        recommended_books = user_recs_exploded.select(col("User-ID"), col("recommendation.Book-Id").alias(
            "Book-Id"), col("recommendation.rating").alias("Predicted-Rating"))

        recommended_books_with_names = recommended_books.join(
            self.books_df, on="Book-Id", how="inner")

        # Scale the ratings to be on [0, 10]
        scaled_recommendations = self.__scale_ratings(
            recommended_books_with_names)

        return self.__convert_df_to_list(scaled_recommendations)

    def __convert_df_to_list(self, df):

        truncated_df = df.withColumn(
            "description", expr("substring(description, 1, 200)"))

        formatted_df = truncated_df.withColumn("Predicted-Rating", format_number(col("Predicted-Rating"), 2)) \
            .withColumn("Scaled-Rating", format_number(col("Scaled-Rating"), 2))

        # Get only the important collumns
        selected_df = formatted_df.select("Book-Title", "Predicted-Rating",
                                          "Scaled-Rating", "categories", "description")

        # make the DF into raw row objects and make them a list of lists
        rows = selected_df.collect()
        list_of_lists = [list(row) for row in rows]

        header = ["Book-Title", "Predicted-Rating",
                  "Scaled-Rating", "Categories", "Description"]
        list_of_lists.insert(0, header)

        list_cropped = []

        for elem in list_of_lists:
            if len(elem[2]) > 300:
                elem[2] = elem[2][:300] + "..."
            list_cropped.append(elem)

        return list_of_lists

    def __scale_ratings(self, predictions):
        # Get the maximum and minimum predicted ratings
        max_rating = predictions.selectExpr(
            "max(`Predicted-Rating`)").collect()[0][0]
        min_rating = 1

        # Scale the ratings to the range [0, 10]
        scaled_predictions = predictions.withColumn(
            "Scaled-Rating", ((col("Predicted-Rating") - min_rating) / (max_rating - min_rating)) * 10)

        # Reorder the columns to place Scaled-Rating next to Predicted-Rating
        scaled_predictions = scaled_predictions.select(
            "User-ID", "Book-Id", "Predicted-Rating", "Scaled-Rating",
            *[col for col in scaled_predictions.columns if col not in ["User-ID", "Book-Id", "Predicted-Rating", "Scaled-Rating"]]
        )

        return scaled_predictions
