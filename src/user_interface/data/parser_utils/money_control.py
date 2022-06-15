"""
@File : money_control.py
@Author : Sudharshana VP
@Contact : sudharshanavp@outlook.com
"""

from typing import List

from parse_utility import get_page

URL_MONEY_CONTROL = "https://www.moneycontrol.com/news/tags/"


def get_page_links(stock_name: str) -> List[str]:
    """Get the moneycontrol website links for the news webpages of a particular stock

    Args:
        stock_name (str): Name of the stock

    Returns:
        List[str]: returns a list of news webpage urls
    """
    base_url = URL_MONEY_CONTROL + stock_name.replace(" ", "-") + ".html"
    number_of_pages = int(
        get_page(base_url).find_all("a", class_="last")[-1]["data-page"]
    )
    page_links = []
    for i in range(1, number_of_pages + 1):
        page_links.append(base_url + "/page-" + str(i) + "/")
    return page_links


def get_article_list(page_links: List[str]) -> List[str]:
    article_list = []
    for link in page_links:
        article_list.append(get_page(link).find("ul", id="cagetory").find_all("a"))
    return sum(article_list, [])


def get_headlines(article_list: List[str]) -> List[str]:
    """Get all headlines available from webpages specified in the input

    Args:
        page_links (List[str]): List of html hyperlink tags(<a> tags) to articles

    Returns:
        List[str]: List of headlines scraped from the articles
    """
    headline_list = []
    for article_block in article_list:
        headline_list.append(article_block.text)
    headline_list = list(filter(("").__ne__, headline_list))
    return headline_list


def get_article_links(article_list: List[str]) -> List[str]:
    article_links = []
    for article_block in article_list:
        link = article_block["href"]
        if link not in article_links:
            article_links.append(link)
    return article_links
