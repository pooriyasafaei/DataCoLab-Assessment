from data_processor.provider import BookData
from data_processor.generator import BookImageGenerator
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import pandas as pd

import math
from string import printable
import re


class BookDataAnalyzer(BookData):
    def __init__(self, table_name='book_summaries_modified', db_name='book_data.db',
                 summarization_model="google/pegasus-xsum"):
        super().__init__()
        self.table_name = table_name
        self.db_name = db_name
        self.df = None
        self.load_from_db(db_name=db_name)
        self.tokenizer = PegasusTokenizer.from_pretrained(summarization_model)
        self.summarizer = PegasusForConditionalGeneration.from_pretrained(summarization_model)
        self.generator = BookImageGenerator()

    def clean_text(self, text):
        text = re.sub(r'\s+', ' ', text)
        text = ''.join(filter(lambda x: x in set(printable), text))
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()

    def summarize_large_text(self, text, max_length=200, chunk_size=500, max_itr=10):
        i = 0
        preferred_max_itr = math.log10(len(text)) + 3
        while len(text) > max_length and i < max_itr and i < preferred_max_itr:
            tokens = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=chunk_size)
            chunks = [tokens.input_ids[i:i + chunk_size]
                      for i in range(0, tokens.input_ids.size(1), chunk_size)]
            summarized_text = ""

            for chunk in chunks:
                summary_ids = self.summarizer.generate(chunk, max_length=max_length,
                                                       num_beams=4, early_stopping=True)
                summary_part = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                summarized_text += summary_part + ' '

            text = summarized_text.strip()
            i += 1
        return text

    def summarize_all_records(self, batch_size=10):
        if 'Condensed Summary' not in self.df.columns:
            self.df['Condensed Summary'] = pd.NA

        total = len(self.df)

        for index, row in self.df.iterrows():
            if pd.isna(row['Condensed Summary']):
                self.condense_text(row, index)
                if (index + 1) % batch_size == 0 or index + 1 == total:
                    print(f"Summarization progress: {index + 1}/{total} records processed.")
        self.save_to_db(table_name='book_summaries_with_condensed')

    def summarize_text(self, text, max_length=200):
        input_ids = self.tokenizer.encode(text, return_tensors="pt")
        summary_ids = self.summarizer.generate(input_ids, max_length=max_length,
                                               num_beams=4, early_stopping=True)
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    def condense_text(self, row, index):
        text = row['Summary']
        if len(text) < 60:
            summary_result = ""
        elif len(text) < 80:
            summary_result = text
        elif len(text) < 1500:
            summary_result = self.summarize_text(text)
        else:
            summary_result = self.summarize_large_text(text)

        if summary_result:
            self.df.at[index, 'Condensed Summary'] = self.clean_text(summary_result)
            row = self.df.loc[index]
            self.update_db_record(row)
            return row
        else:
            print(f"No summary generated for index {index}, check input data.")

    def generate_image_for_random_samples(self, sample_size=10, inference_steps=100, guidance_scale=8):
        sample_df = self.df.sample(n=sample_size)
        for index, row in sample_df.iterrows():
            print(f"Condensing summary for the '{row['Title']}' Book. \n"
                  f"Summary len: {row['Summary Length']}")
            self.generator.inference_steps = inference_steps
            self.generator.guidance_scale = guidance_scale
            row = self.condense_text(row, index)
            image, prompt = self.generator.process_book_record(row)
            filename = f"{row['Title']}_{inference_steps}_{int(guidance_scale)}.png".replace('/', '_')
            self.generator.save_image(image, path="../images/",
                                      filename=filename)
            self.generator.log_image_generation(prompt, filename,
                                                row['Title'], row['URL'])
            print(f"Image generated for the '{row['Title']}' Book.")
