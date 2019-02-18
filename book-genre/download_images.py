#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import codecs
import pandas as pd
from urllib import request
from tqdm import trange
from joblib import Parallel, delayed

OUTPUT_DIRPATH="data/images"
CSV_FILEPATH="data/book-data.csv"

header_names = ['Amazon ID (ASIN)', 'Filename', 'Image URL', 'Title', 'Author', 'Category ID',
                'Category']
with codecs.open(CSV_FILEPATH, mode='r', encoding='utf-8', errors='ignore') as f:
    csv = pd.read_csv(f, delimiter=",", header=None, names=header_names)

if not os.path.isdir(OUTPUT_DIRPATH):
    os.makedirs(OUTPUT_DIRPATH)

print('[Download images into "{}"]'.format(OUTPUT_DIRPATH))

def download_image(i):
    filename = csv.iloc[i]['Filename']
    category = csv.iloc[i]['Category']
    inner_output_dirpath = os.path.join(OUTPUT_DIRPATH, category)
    if not os.path.isdir(inner_output_dirpath):
        os.mkdir(inner_output_dirpath)
    output_filepath = os.path.join(inner_output_dirpath, filename)

    url = csv.iloc[i]['Image URL']
    if not os.path.isfile(output_filepath):
        downloaded_img = request.urlopen(url)
        f = open(output_filepath, mode='wb')
        f.write(downloaded_img.read())
        downloaded_img.close()
        f.close()

Parallel(n_jobs=-1)(delayed(download_image)(i) for i in range(len(csv)))