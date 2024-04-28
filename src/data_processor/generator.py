import json
import os
import sqlite3

import torch
from diffusers import StableDiffusionPipeline


class BookImageGenerator:
    def __init__(self, model_path="CompVis/stable-diffusion-v1-4", device='cuda',
                 inference_steps=50, guidance_scale=8):
        # Move the pipeline to GPU if Nvidia driver is available
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = StableDiffusionPipeline.from_pretrained(model_path).to(self.device)
        self.inference_steps = inference_steps
        self.guidance_scale = guidance_scale

    def create_prompt(self, title, summary, author=None, genres=None, year=None):
        try:
            genres = json.loads(genres)
            genre_list = [genre_name for genre_id, genre_name in genres.items()]
            genre_text = ', '.join(genre_list).lower() if genre_list else ""
        except Exception:
            genre_text = ""

        prompt = f"{title}: "
        try:
            year = int(year)
            prompt += f"A book published in {year}, "
        except ValueError:
            pass

        if len(genre_text) > 0:
            prompt += f"categorized under {genre_text}"

        if len(author) > 0:
            prompt += f" and written by {author}"

        if len(summary) > 0:
            prompt += f". The story: {summary}."

        return prompt

    def generate_image(self, prompt):
        with torch.no_grad():
            print(f"Generating image for prompt: {prompt}")
            image = self.model(prompt,
                               num_inference_steps=self.inference_steps,
                               guidance_scale=self.guidance_scale
                               ).images[0]
            print("Image generated successfully.")
        return image

    def process_book_record(self, record):
        title = record['Title']
        summary = record['Condensed Summary']
        author = record['Author']
        genres = record['Genres']
        year = record['Publication Year']

        prompt = self.create_prompt(title, summary, author, genres, year)
        image = self.generate_image(prompt)
        return [image, prompt]

    def save_image(self, image, path="../images/", filename="output_image.png"):
        if not os.path.exists(path):
            os.makedirs(path)

        full_path = os.path.join(path, filename)
        image.save(full_path)
        print(f"Image saved to {full_path}")

    def log_image_generation(self, prompt, filename, book_title, book_url, db_name='book_data.db'):
        """Log the image generation event in the database."""
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        self.create_log_table(db_name)
        cursor.execute("""
            INSERT INTO ImageGenerationLog (Prompt, ImageFilename, BookTitle, BookURL)
            VALUES (?, ?, ?, ?);
        """, (prompt, filename, book_title, book_url))
        conn.commit()
        conn.close()

    def create_log_table(self, db_name='book_data.db'):
        """Create a table for logging image generation events if it doesn't exist."""
        conn = None
        try:
            conn = sqlite3.connect(db_name)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ImageGenerationLog (
                    ID INTEGER PRIMARY KEY AUTOINCREMENT,
                    Prompt TEXT,
                    ImageFilename TEXT,
                    BookTitle TEXT,
                    BookURL TEXT
                );
            """)
            conn.commit()
        except sqlite3.Error as e:
            print(f"An error occurred while creating the logging table: {e}")
        finally:
            if conn:
                conn.close()
