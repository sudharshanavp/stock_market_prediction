'''
@File : parse_utility.py
@Author : Sudharshana VP
@Contact : sudharshanavp@protonmail.com
'''

import re
from typing import Optional, List
from urllib import response
from bs4 import BeautifulSoup
from requests import get

import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

stopwords = stopwords.words('english')

def get_page(url_string: str) -> BeautifulSoup:
    response = get(url_string)
    return BeautifulSoup(response.text, 'html.parser')

def clean_article(article: str) -> List[str]:
    article = article.replace('/(\n)/gm', " ")
    article = re.sub('[.,!?:;%&$^*@#)/(-''`"—=+]', ' ', article)
    article = re.sub('[0-9]', ' ', article)
    article = article.replace("`|’|”|“", "'")
    article = article.replace("/(\\x)/g", "")
    new_stop_words = ["said", "also", "per", "cent", "would", "last", "first", "like", '\'', '\"', "\'", "\"", "’", "'s", "“", "”"]
    stopwords.extend(new_stop_words)
    clean = [word for word in word_tokenize(article) if not word in stopwords]
    return clean

def lower_case(article: str) -> List[str]:
    lower = [word.casefold() for word in article]