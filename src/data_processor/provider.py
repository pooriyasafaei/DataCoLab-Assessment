import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import json


class BookDataAnalysis:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.genre_dict = {}

    def parse_genres(self):
        for genre_str in self.df['Genres']:
            try:
                genre_dict = json.loads(
                    genre_str.replace("'", '"'))  # Replacing single quotes with double quotes for JSON parsing
                for url, genre in genre_dict.items():
                    if url not in self.genre_dict:
                        self.genre_dict[url] = genre
            except json.JSONDecodeError:
                continue

    def load_data(self):
        with open(self.file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        data = []
        for line in lines:
            parts = line.strip().split('\t')
            data.append(parts)

        columns = ['ID', 'URL', 'Title', 'Author', 'Publication Date', 'Genres', 'Summary']
        self.df = pd.DataFrame(data, columns=columns)

        na_rows = self.df[self.df.isnull().any(axis=1)]
        if not na_rows.empty:
            print("Rows with missing values:")
            print(na_rows)

        self.save_to_db(table_name='book_summaries_raw')

    def explore_data(self):
        self.df['Publication Date'] = pd.to_datetime(self.df['Publication Date'], errors='coerce', format='%Y-%m-%d')

        plt.figure(figsize=(10, 5))
        self.df['Publication Year'] = self.df['Publication Date'].dt.year
        self.df['Publication Year'].dropna().astype(int).hist(bins=30, color='blue', edgecolor='black')
        plt.title('Distribution of Publication Years')
        plt.xlabel('Year')
        plt.ylabel('Number of Books')
        plt.show()

        plt.figure(figsize=(10, 5))
        self.df['Summary Length'] = self.df['Summary'].apply(len)
        self.df['Summary Length'].hist(bins=50, color='green', edgecolor='black')
        plt.title('Distribution of Summary Lengths')
        plt.xlabel('Length of Summary')
        plt.ylabel('Number of Summaries')
        plt.show()

    def get_dataframe(self):
        return self.df

    def save_to_db(self, db_name='book_data.db', table_name='book_summaries'):
        conn = sqlite3.connect(db_name)
        cur = conn.cursor()

        self.df.to_sql(table_name, conn, if_exists='replace', index=False)

        conn.close()
        print(f"Data saved to {db_name} in table {table_name}.")
