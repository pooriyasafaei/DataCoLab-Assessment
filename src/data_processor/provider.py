import os
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import logging
from wordcloud import WordCloud

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class BookData:
    def __init__(self):
        self.df = None

    def save_to_db(self, db_name='book_data.db', table_name='book_summaries'):
        try:
            conn = sqlite3.connect(db_name)
            self.df.to_sql(table_name, conn, if_exists='replace', index=False)
            conn.close()
            logging.info(f"Data saved to {db_name} in table {table_name}.")
        except sqlite3.Error as e:
            logging.error(f"An error occurred while saving data to the database: {e}")

    def load_from_db(self, db_name='book_data.db', table_name='book_summaries_modified'):
        try:
            conn = sqlite3.connect(db_name)
            self.df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
            conn.close()
            print(f"Data loaded from {db_name}, table {table_name}.")
        except Exception as e:
            print(f"Failed to load data from database: {e}")

    def update_db_record(self, row, db_name='book_data.db', table_name='book_summaries_modified'):
        columns = row.index.tolist()
        values = [row[col] for col in columns if col != 'URL']
        set_clause = ", ".join([f'"{col}" = ?' for col in columns if col != 'URL' and col != 'Condensed Summary'])
        sql = f'UPDATE "{table_name}" SET {set_clause} WHERE URL = ?'
        values.append(row['URL'])

        conn = None
        try:
            conn = sqlite3.connect(db_name)
            cursor = conn.cursor()
            cursor.execute(sql, values)
            conn.commit()
            print(f"Record updated successfully: URL {row['URL']}")
        except sqlite3.Error as e:
            print(f"Database error: {e}")
        finally:
            if conn:
                conn.close()


class BookDataProvider(BookData):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def load_data_file(self):
        with open(self.file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        data = []
        for line in lines:
            parts = line.strip().split('\t')
            parts[4] = parts[4].split('-')[0]
            data.append(parts)

        columns = ['ID', 'URL', 'Title', 'Author', 'Publication Date', 'Genres', 'Summary']

        self.df = pd.DataFrame(data, columns=columns)

        na_rows = self.df[self.df.isnull().any(axis=1)]
        if not na_rows.empty:
            print("Rows with missing values:")
            print(na_rows)

        self.df['Publication Date'] = pd.to_datetime(self.df['Publication Date'], errors='coerce', format='%Y')
        self.df['Publication Year'] = self.df['Publication Date'].dt.year

        self.save_to_db(table_name='book_summaries_raw')

    def explore_data(self):
        # Distribution of Publication Years
        plt.figure(figsize=(10, 5))
        self.df['Publication Year'].dropna().astype(int).hist(bins=30, color='blue', edgecolor='black')
        plt.title('Distribution of Publication Years')
        plt.xlabel('Year')
        plt.ylabel('Number of Books')
        self.save_plot(plt, 'publication_years.png')
        plt.show()

        # Distribution of Summary Lengths
        plt.figure(figsize=(10, 5))
        self.df['Summary Length'] = self.df['Summary'].apply(len)
        self.df['Summary Length'].hist(bins=50, color='green', edgecolor='black')
        plt.title('Distribution of Summary Lengths')
        plt.xlabel('Length of Summary')
        plt.ylabel('Number of Summaries')
        self.save_plot(plt, 'summary_lengths.png')
        plt.show()

        # Genre Distribution
        plt.figure(figsize=(10, 5))
        genres = self.df['Genres']
        genre_counts = genres.value_counts()
        genre_counts.plot(kind='bar', color='purple')
        plt.title('Genre Distribution')
        plt.xlabel('Genre')
        plt.ylabel('Frequency')
        self.save_plot(plt, 'genre_distribution.png')
        plt.show()

        # Word Cloud for Summaries
        plt.figure(figsize=(10, 5))
        text = ' '.join(self.df['Summary'].dropna())
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud of Book Summaries')
        self.save_plot(plt, 'word_cloud.png')
        plt.show()

    def fill_missing_dates(self):
        avg_years = self.df.groupby('Author')['Publication Year'].transform('mean').round(0)
        self.df.loc[self.df['Publication Year'].isnull(), 'Publication Year'] = avg_years
        self.df['Publication Year'] = self.df['Publication Year'].where(self.df['Author'].notna(), other=None)
        self.save_to_db(table_name='book_summaries_modified')

    def get_dataframe(self):
        return self.df

    def save_plot(self, plt, file_name):
        if not os.path.exists('../plots'):
            os.makedirs('../plots')
        plt.savefig(f"../plots/{file_name}")
