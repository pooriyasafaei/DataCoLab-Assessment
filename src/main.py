from data_processor.downloader import DriveDownloader
from data_processor.provider import BookDataProvider
from data_processor.analyzer import BookDataAnalyzer
import os

data_root = '../data/'


def download_data(file_id, file_name):
    destination_path = data_root + file_name
    if not os.path.exists(destination_path):
        dler = DriveDownloader(file_id, file_name)
        dler.download()
    else:
        print("data file already exists.")


if __name__ == '__main__':
    data_file_name = 'booksummaries.txt'
    download_data('1PDujXAkSelYQ8KX_vBGwDr6EkMhJXlA3', data_file_name)
    data_provider = BookDataProvider(data_root + data_file_name)
    data_provider.load_data_file()
    data_provider.explore_data()
    # data_provider.save_to_db(table_name='book_summaries_analytics')
    # data_provider.fill_missing_dates()
    # mean_summary_len = data_provider.get_dataframe()['Summary Length'].mean()
    # print(mean_summary_len)
    # data_analyzer = BookDataAnalyzer()
    # data_analyzer.generate_image_for_random_samples(sample_size=30, inference_steps=50, guidance_scale=8)
