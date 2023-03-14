import time

import requests
from bs4 import BeautifulSoup
import re
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from selenium import webdriver

# Downloading Stopwords and Master Dictionary
nltk.download('stopwords')
nltk.download('punkt')

import requests
import os

# Downloading the file locally
url_stopwords = 'https://raw.githubusercontent.com/Alir3z4/stop-words/master/english.txt'
stopwords_file = 'english.txt'
if not os.path.isfile(stopwords_file):
    stopwords_response = requests.get(url_stopwords)
    with open(stopwords_file, 'w') as f:
        f.write(stopwords_response.text)

url_master_dict = 'https://raw.githubusercontent.com/jeffreybreen/twitter-sentiment-analysis-tutorial-201107/master/data/opinion-lexicon-English/'
master_dict_positive_file = 'positive-words.txt'
master_dict_negative_file = 'negative-words.txt'
if not os.path.isfile(master_dict_positive_file):
    master_dict_positive_response = requests.get(url_master_dict + 'positive-words.txt')
    with open(master_dict_positive_file, 'w') as f:
        f.write(master_dict_positive_response.text)

if not os.path.isfile(master_dict_negative_file):
    master_dict_negative_response = requests.get(url_master_dict + 'negative-words.txt')
    with open(master_dict_negative_file, 'w') as f:
        f.write(master_dict_negative_response.text)

# Creating Stopwords List and Positive/Negative Dictionary
with open(stopwords_file, 'r') as f:
    stopwords_list = set(stopwords.words('english') + f.read().split())

with open(master_dict_positive_file, 'r') as f:
    master_dict_positive = set(f.read().split())

with open(master_dict_negative_file, 'r') as f:
    master_dict_negative = set(f.read().split())


# Regex pattern for Personal Pronouns
pronoun_pattern = re.compile(r"\b(I|we|my|ours|us)\b", flags=re.IGNORECASE)

# # Read the CSV file containing the article URLs
# df = pd.read_excel('Input.xlsx', sheet_name='Sheet1')



def get_clean_text(article_url):
    driver = webdriver.Chrome()
    url = article_url
    driver.maximize_window()
    driver.get(url)
    content = driver.page_source.encode('utf-8').strip()
    soup = BeautifulSoup(content,"html.parser")
    content = soup.find('div', class_='td-post-content')
    content2 = soup.find('h1',class_='entry-title')
    text = content2.get_text()+'\n'+content.get_text()
    print(text)
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuations
    text = text.lower()  # Convert to lowercase
    words = word_tokenize(text)
    words_cleaned = [word for word in words if word not in stopwords_list]
    text_cleaned = ' '.join(words_cleaned)
    return text_cleaned


# # Get the cleaned text for each article URL
# cleaned_texts = []
# for url in df['URL']:
#     cleaned_text = get_clean_text(url)
#     cleaned_texts.append(cleaned_text)


# Function to calculate Positive Score
def get_positive_score(text):
    positive_words = master_dict_positive - stopwords_list
    positive_count = sum(1 for word in word_tokenize(text) if word in positive_words)
    return positive_count


# Function to calculate Negative Score
def get_negative_score(text):
    negative_words = master_dict_negative - stopwords_list
    negative_count = sum(1 for word in word_tokenize(text) if word in negative_words)
    return -1 * negative_count


# Function to calculate Polarity Score
def get_polarity_score(text):
    positive_score = get_positive_score(text)
    negative_score = get_negative_score(text)
    polarity_score = (positive_score - negative_score) / (positive_score + negative_score + 0.000001)
    return polarity_score


# Function to calculate Subjectivity Score
def get_subjectivity_score(text):
    positive_score = get_positive_score(text)
    negative_score = get_negative_score(text)
    total_words = len(word_tokenize(text))
    subjectivity_score = (positive_score + negative_score) / (total_words + 0.000001)
    return subjectivity_score


# Function to calculate Average Sentence Length
def get_avg_sentence_length(text):
    sentences = sent_tokenize(text)
    num_sentences = len(sentences)
    num_words = len(word_tokenize(text))
    avg_sentence_length = num_words / num_sentences
    return avg_sentence_length


# Function to calculate Percentage of Complex Words

def get_percentage_complex_words(text):
    words = word_tokenize(text)
    num_complex_words = 0
    for word in words:
        if len(word) > 2 and pronoun_pattern.match(word) is None:
            num_complex_words += 1
            percentage_complex_words = (num_complex_words / len(words)) * 100
            return percentage_complex_words


def get_fog_index(text):
    avg_sentence_length = get_avg_sentence_length(text)
    percentage_complex_words = get_percentage_complex_words(text)
    fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)
    return fog_index


articles = pd.read_excel('Input.xlsx', sheet_name='Sheet1')
# #getting only first 2 rows
# articles = articles.head(2)
articles['clean_text'] = articles['URL'].apply(get_clean_text)
articles['polarity_score'] = articles['clean_text'].apply(get_polarity_score)
articles['subjectivity_score'] = articles['clean_text'].apply(get_subjectivity_score)
articles['avg_sentence_length'] = articles['clean_text'].apply(get_avg_sentence_length)
articles['percentage_complex_words'] = articles['clean_text'].apply(get_percentage_complex_words)
articles['fog_index'] = articles['clean_text'].apply(get_fog_index)
# dropping clean text
articles = articles.drop(['clean_text'], axis=1)
articles.to_csv('articles_sentiment_scores.csv',index=False)