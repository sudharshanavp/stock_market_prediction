"""
@File : parse_utility.py
@Author : Sudharshana VP
@Contact : sudharshanavp@outlook.com
"""

import re
from typing import List, NoReturn
from bs4 import BeautifulSoup
from requests import get

import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stopwords = stopwords.words("english")


def get_page(url_string: str) -> BeautifulSoup:
    """Scrape the page specified by the input url

    Args:
        url_string (str): input url

    Returns:
        BeautifulSoup: return the page obtained
    """
    response = get(url_string)
    return BeautifulSoup(response.text, "html.parser")


def clean_article(article: str) -> List[str]:
    """Clean the article by removing punctuations and stop words

    Args:
        article (str): article content in string format

    Returns:
        List[str]: list of words in the article
    """
    article = article.replace("/(\n)/gm", " ")
    article = re.sub("[.,!?:;%&$^*@#)/(-" '`"—=+]', " ", article)
    article = re.sub("[0-9]", " ", article)
    article = article.replace("`|’|”|“", "'")
    article = article.replace("/(\\x)/g", "")
    new_stop_words = [
        "said",
        "also",
        "per",
        "cent",
        "would",
        "last",
        "first",
        "like",
        "'",
        '"',
        "'",
        '"',
        "’",
        "'s",
        "“",
        "”",
        " Ltd.",
    ]
    stopwords.extend(new_stop_words)
    clean = [
        word for word in word_tokenize(article) if not word in stopwords  # noqa: E713
    ]
    return clean


def lower_case(article: List[str]) -> List[str]:
    """Convert all characters of the words in the article to lowercase

    Args:
        article (List[str]): words present in the article

    Returns:
        List[str]: words in the article converted to lowercase
    """
    lower = [word.casefold() for word in article]
    return lower


def sort_dictionary(dictionary: dict) -> dict:
    """Sort the input dictionary

    Args:
        dictionary (dict): input dictionary

    Returns:
        dict: sorted dictionary
    """
    sorted_dict = dict(sorted(dictionary.items(), key=lambda value: value[1]))
    return dict(reversed(list(sorted_dict.items())))


def export_as_csv(dictionary: dict) -> NoReturn:
    """Export dictionary as csv file

    Args:
        dictionary (dict): input dictionary

    Returns:
        NoReturn: Export the csv file to data folder in project root directory
    """
    export = {"word": list(dictionary.keys()), "frequency": list(dictionary.values())}
    export_dataframe = pd.DataFrame(export)
    export_dataframe.to_csv("../../data/word_frequency.csv")


def update_frequency(element: str, dictionary: dict) -> dict:
    """Update frequency of each word in the dictionary

    Args:
        element (str): word to be updated
        dictionary (dict): the dictionary containing words and their frequency count

    Returns:
        dict: updated dictionary
    """
    if element in dictionary:
        dictionary[element] += 1
    else:
        dictionary.update({element: 1})
    return dictionary


def get_word_frequency(article_list: List[str]) -> NoReturn:
    """Find the frequency of words in the article

    Args:
        article_list (List[str]): list of articles parsed from the webpage

    Returns:
        NoReturn: export word frequncy as a csv file
    """
    word_frequency = {}
    for article in article_list:
        article = article.replace("/\r?\n|\r/g", " ")

    for article in article_list:
        cleaned_article = clean_article(article["article"])
        cleaned_article = lower_case(cleaned_article)

        for word in cleaned_article:
            if len(word) > 2:
                word_frequency = update_frequency(word, word_frequency)

    word_frequency_sorted = sort_dictionary(dict(word_frequency))
    export_as_csv(word_frequency_sorted)
