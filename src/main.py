from data_processor.downloader import DriveDownloader
from data_processor.provider import BookDataAnalysis
import os

data_root = '../data/'


def download_data(file_id, file_name):
    destination_path = data_root + file_name
    if not os.path.exists(destination_path):
        dler = DriveDownloader(file_id, file_name)
        dler.download()
    else:
        print("File already exists.")


if __name__ == '__main__':
    data_file_name = 'booksummaries.txt'
    download_data('1PDujXAkSelYQ8KX_vBGwDr6EkMhJXlA3', 'booksummaries.txt')
    data_analysis = BookDataAnalysis(data_root + data_file_name)
    data_analysis.load_data()
    data_analysis.explore_data()