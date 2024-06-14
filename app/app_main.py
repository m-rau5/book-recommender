from customtkinter import *
from user_profiles.users import User_4017, User_6238, User_62891
from CTkTable import CTkTable
from scripts.collab_filtering import Collab_Recommender
from scripts.content_based import Content_Recommender
from tkinter import messagebox

# dict for the button actions
user_lists = {
    "User_4017": User_4017,
    "User_6238": User_6238,
    "User_62891": User_62891
}

book_dataset = "app/datasets/final_books.csv"
ratings_dataset = "app/datasets/final_ratings.csv"
als_model = "C:\\Users\\Raul\\Documents\\python prj\\theisis\\book-recommender - Testing\\app\\models\\collab_filtering\\ALS_50_100_07"
tokenized_books = "app\datasets\\tokenized_books_df.csv"
w2v_model = "app\models\content_based\word2vec.model"

# Flags for table creation (if 0) or rebuilding (if 1)
collab_recommender_flag = 0
content_recommender_flag = 0


class BookRecommenderApp(CTk):
    def __init__(self):
        super().__init__()
        self.title('Book Recommender Tool')
        # self.resizable(0, 0)
        self.geometry("1280x720")
        set_appearance_mode("light")

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.tab_view = TabView(master=self)
        self.tab_view.grid(row=0, column=0, padx=20, pady=20, sticky="news")


class TabView(CTkTabview):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self.add("Collaborative Filtering")  # add tab at the end
        self.add("Content Based")  # add tab at the end
        self.set("Collaborative Filtering")  # set currently visible tab

        # add widgets on tabs
        self.label = Collab_Filtering(
            master=self.tab("Collaborative Filtering"))
        self.label.pack(expand=True, fill="both")

        self.label = Content_Based(
            master=self.tab("Content Based"))
        self.label.pack(expand=True, fill="both")


class Collab_Filtering(CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.collab_brs = Collab_Recommender(
            book_dataset, ratings_dataset, als_model)

        self.user_id = "User_4017"

        # User selection frame
        self.user_frame = CTkFrame(master=self)
        self.user_frame.pack(pady=(10, 0))

        # User Selection
        self.user_select = CTkComboBox(
            master=self.user_frame, values=["User 4017", "User 6238", "User 62891"])
        self.user_select.pack()

        self.button = CTkButton(
            master=self, text="Select this User", command=self.Show_user_books)
        self.button.pack(pady=(15, 10))

        # Table UI for users:
        self.user_table_label = CTkLabel(
            master=self, text="These are all the books reviewed by the user:")
        self.user_table_label.pack(pady=(0, 5))

        # Create the frame of the table
        self.user_table_frame = CTkScrollableFrame(master=self)
        self.user_table_frame.pack(fill="x")

        self.user_table = CTkTable(
            master=self.user_table_frame, column=3, values=User_4017, wraplength=900)
        self.user_table.pack(fill="x")

        # Button for recommending books
        self.button = CTkButton(
            master=self, text="Recommend Books!", command=self.Recommend_books_for_user)
        self.button.pack(pady=(15, 10))

    def Show_user_books(self):
        self.user_id = self.user_select.get()
        self.user_id = self.user_id.replace(" ", "_")
        selected_user = user_lists[self.user_id]
        self.user_table.destroy()  # Destroy the existing table
        self.user_table = CTkTable(
            master=self.user_table_frame, column=3, values=selected_user, wraplength=900, justify=CENTER)
        self.user_table.pack(fill="x")
        self.user_table_frame.update()
        self.user_table_frame.update_idletasks()

    def Recommend_books_for_user(self):

        user_id = self.user_id[5:]

        global collab_recommender_flag

        if (collab_recommender_flag == 0):
            # Create the frame of the recommendations table if not created
            self.recommender_table_frame = CTkScrollableFrame(master=self)
            self.recommender_table_frame.pack(expand=True, fill="both")
            collab_recommender_flag = 1
        else:
            # If table is getting regenerated, just destroy the old one
            self.recommender_table.destroy()

        user_recommended_df = self.collab_brs.get_recommendations(user_id)
        self.recommender_table = CTkTable(
            master=self.recommender_table_frame, column=5, values=user_recommended_df, wraplength=300)
        self.recommender_table.pack(expand=True, fill="both")


class Content_Based(CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self.content_BRS = Content_Recommender(tokenized_books, w2v_model)

        self.algo_label = CTkLabel(master=self, text="Pick an Algorithm:")
        self.algo_label.pack(pady=(10, 0))

        self.algo_select = CTkComboBox(
            master=self, values=["TF-IDF", "Count Vectorizer", "Word2Vec"])
        self.algo_select.pack(pady=(5, 0))

        self.book_label = CTkLabel(master=self, text="Pick a Book:")
        self.book_label.pack(pady=(10, 0))

        self.book_select = CTkComboBox(master=self, values=[
            "The Lord of the Rings", "Les Miserables (Penguin Classics)", "Midnight Voices"], width=250)
        self.book_select.pack(pady=(5, 10))

        self.recommend_button = CTkButton(
            master=self, text="Recommend Book", command=self.Recommend_Content)
        self.recommend_button.pack(pady=(5, 15))

    def Recommend_Content(self):
        algo = self.algo_select.get()
        book = self.book_select.get()

        if book not in self.book_select._values:
            messagebox.showwarning(
                "Warning", "Please select a book")
            return

        if algo == "TF-IDF":
            content = self.content_BRS.TF_IDF(book)
        elif algo == "Count Vectorizer":
            content = self.content_BRS.Count_Vectorizer(book)
        elif algo == "Word2Vec":
            content = self.content_BRS.Word_2_Vec(book)
        else:
            messagebox.showwarning(
                "Warning", "Please select a valid algorithm")
            return

        global content_recommender_flag

        if (content_recommender_flag == 0):
            # Create the frame of the recommendations table if not created
            self.recommender_table_frame = CTkScrollableFrame(master=self)
            self.recommender_table_frame.pack(expand=True, fill="both")
            content_recommender_flag = 1
        else:
            # If table is getting regenerated, just destroy the old one
            self.recommender_table.destroy()

        self.recommender_table = CTkTable(
            master=self.recommender_table_frame, column=3, values=content, wraplength=400)
        self.recommender_table.pack(expand=True, fill="both")


if __name__ == "__main__":
    app = BookRecommenderApp()
    app.mainloop()
