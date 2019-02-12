import csv
import json
import random

TRUNCATE_SIZE = 64
FILL_CARACHTER = '|'
FILLING = FILL_CARACHTER * TRUNCATE_SIZE

book_data = []
genres = {}
with open('data/book-data.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)

    for row in csv_reader:
        book_data.append(((f"{row[4]} // {row[3]}" + FILLING)[:TRUNCATE_SIZE], row[5]))

        if row[6] not in genres:
            genres[row[6]] = int(row[5])
random.shuffle(book_data)

with open('data/parsed-book-data.csv', 'w') as csv_file:
    csv_writer = csv.writer(csv_file)

    for book in book_data:
        csv_writer.writerow([book[0], book[1]])

with open('data/genres.json', 'w') as json_file:
    json.dump(genres, json_file)