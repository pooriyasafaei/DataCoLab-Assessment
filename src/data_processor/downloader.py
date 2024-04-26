import requests
import os


class DriveDownloader:
    def __init__(self, file_id, file_name, destination="../data/", chunk_size=32768):
        self.file_id = file_id
        self.destination = destination + file_name
        self.session = requests.Session()
        self.base_url = "https://drive.google.com/uc?export=download"
        self._prepare_directory()
        self.chunk_size = chunk_size

    def download(self):
        try:
            response = self.session.get(f"{self.base_url}&id={self.file_id}", stream=True)
            token = self._get_confirm_token(response)

            if token:
                params = {'id': self.file_id, 'confirm': token}
                response = self.session.get(self.base_url, params=params, stream=True)

            self._save_response_content(response)
            print("Data downloaded successfully!")
        except Exception as e:
            print(f"An error occurred: {e}")

    def _get_confirm_token(self, response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def _save_response_content(self, response):
        with open(self.destination, "wb") as f:
            for chunk in response.iter_content(self.chunk_size):
                if chunk:
                    f.write(chunk)

    def _prepare_directory(self):
        os.makedirs(os.path.dirname(self.destination), exist_ok=True)
