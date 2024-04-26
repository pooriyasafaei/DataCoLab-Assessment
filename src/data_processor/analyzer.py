import pandas as pd
from data_processor.provider import BookData
from transformers import pipeline


class BookDataAnalyzer(BookData):
    def __init__(self, table_name='book_summaries_modified', db_name='book_data.db',
                 summarization_model="facebook/bart-large-cnn"):
        super().__init__()
        self.table_name = table_name
        self.db_name = db_name
        self.df = None
        self.load_from_db(db_name=db_name)
        self.summarizer = pipeline("summarization", model=summarization_model)

    def summarize_texts(self, batch_size=10, static_max_length=120, static_min_length=60, truncation=True):
        if 'Condensed Summary' not in self.df.columns:
            self.df['Condensed Summary'] = pd.NA

        total = len(self.df)

        for index, row in self.df.iterrows():
            if pd.isna(row['Condensed Summary']):
                if len(row['Summary']) < 80:
                    summary_result = row['Summary']
                elif len(row['Summary']) < 1000:
                    try:
                        print(f"Summarizing text for index {index}:\n text: {row['Summary']}")
                        max_len = min(static_max_length, len(row['Summary']))
                        min_len = min(static_min_length, len(row['Summary']))
                        summary_result = self.summarizer(row['Summary'], max_length=max_len,
                                                         min_length=min_len, truncation=truncation)
                    except Exception as e:
                        print(f"Error summarizing text for index {index}: {e}")
                        summary_result = ''
                else:
                    print(f"Large text at index {index}. Applying chunking...")
                    summary_result = self.summarize_large_text(row['Summary'])

                if summary_result:
                    self.df.at[index, 'Condensed Summary'] = summary_result[0]['summary_text']
                else:
                    print(f"No summary generated for index {index}, check input data.")
                if (index + 1) % batch_size == 0 or index + 1 == total:
                    print(f"Summarization progress: {index + 1}/{total} records processed.")

        self.save_to_db(table_name='book_summaries_with_condensed')

    def summarize_text(self, text, max_length=120, min_length=60, truncation=True):
        pass
    def summarize_large_text(self, text, max_length=120, chunk_size=1024):
        while len(text) > max_length:
            chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
            summarized_text = ""

            for chunk in chunks:
                try:
                    summary_part = self.summarizer(chunk, max_length=int(len(chunk) * 0.75), min_length=50,
                                                   truncation=True)
                    if summary_part:
                        summarized_text += summary_part[0]['summary_text'] + ' '
                except Exception as e:
                    print(f"Failed to summarize chunk: {e}")
                    summarized_text += chunk + ' '
            text = summarized_text.strip()

        return text
