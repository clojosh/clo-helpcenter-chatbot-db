import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
import tiktoken
from tqdm import tqdm


def logger(title: str = "", text: str = "") -> None:
    """
    Logs a message.

    Args:
        title (str, optional): The title of the message. Defaults to "".
        text (str, optional): The text of the message. Defaults to "".
    """
    if title and text:
        # Log the title and text in green color
        logging.info("\033[92m" + title + "\033[0m" + "\n" + str(text))
    elif title:
        # Log the title in green color
        logging.info("\033[92m" + title + "\033[0m")
    else:
        # Log the text
        logging.info(str(text))


def num_tokens_from_string(string: str, model: str = "gpt-4o-mini") -> int:
    """Returns the number of tokens in a text string. https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb"""
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def remove_html_tags(article: str) -> str:
    """Remove html tags from an article"""

    article = article.replace("<br>", "\n")

    return re.sub(r"<.*?>", "", article)


def remove_miscellaneous_text(article):
    misc_list = ["Go back to the List of Contents"]

    for misc in misc_list:
        article = article.replace(misc, "")

    # For FAQ articles, remove the Question and word "Answer"
    question_starting_index = article.find("Question")
    answer_ending_index = article.find("Answer")
    if question_starting_index != -1 and answer_ending_index != -1:
        article = article[answer_ending_index + 6 :]

    return article.strip()


def trim_tokens(article):
    """Removes unnecessary tokens"""
    article = article.replace("\u00a0", " ").replace("&nbsp", " ")
    article = re.sub(r"[^\w0-9-\s\n_*.`~!@#$%^&()+={}\:\"'?/><,/+\[\]]", "", article)
    article = re.sub(r"\n+", " ", article)
    article = re.sub(r"\s+", " ", article)

    return article.strip()


def extract_youtube_links(article: str):
    """Return a list of youtube links from an article"""

    iframe_regex = r"<iframe(?:\stitle=\"[a-zA-Z0-9\s]*\")? src=\"[^\s]*\""
    iframes = re.findall(iframe_regex, article)

    youtube_links = []
    for iframe in iframes:
        id = iframe.split("/")[-1].split("?")[0].replace('"', "")
        youtube_links.append("https://www.youtube.com/watch?v={}".format(id))

    return youtube_links


def verify_path(document_path):
    if not os.path.exists("./documents"):
        os.mkdir("./documents")

    if not os.path.exists(document_path):
        os.mkdir(document_path)


def check_create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def sanitize_directory_file_name(text):
    return re.sub(r'[\\/*?:"<>|]', "", text)


def get_section_and_category(env, section_id):
    section_response = requests.request(
        "GET",
        env.get_zendesk_article_section_api_endpoint(section_id),
        headers={
            "Content-Type": "application/json",
        },
    )
    section_objects = json.loads(section_response.text)

    category_response = requests.request(
        "GET",
        env.get_zendesk_article_category_api_endpoint(section_objects["section"]["category_id"]),
        headers={
            "Content-Type": "application/json",
        },
    )
    category_objects = json.loads(category_response.text)

    return (
        section_objects["section"]["id"],
        section_objects["section"]["name"],
        category_objects["category"]["id"],
        category_objects["category"]["name"],
    )


def format_datetime(start_date: str, end_date: str):
    start_date_dt = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S").date()
    end_date_dt = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S").date()

    return str(start_date_dt), str(end_date_dt)


if __name__ == "__main__":
    section_response = requests.request(
        "GET",
        "https://clo3d.zendesk.com/api/v2/help_center/en-us/sections/29864642061593.json",
        headers={
            "Content-Type": "application/json",
        },
    )
    section_objects = json.loads(section_response.text)
    print(section_objects["section"]["name"])
