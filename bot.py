import pandas as pd
import numpy as np
import sklearn

from bs4 import BeautifulSoup
import requests
from ratelimit import limits, sleep_and_retry

import pymorphy2
import os

import urllib

import re

from joblib import dump, load
import pickle

import telegram
from telegram.ext import Updater
from telegram.ext import CommandHandler, CallbackQueryHandler
import logging
from telegram.ext import MessageHandler, Filters
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from dataclasses import dataclass


logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

TOKEN = os.getenv("BOT_TOKEN")

END_BUTTON = "end"

with open("models/tfidf.model", "rb") as fin:
    TFIDF_MODEL = pickle.load(fin)

CLF = load('models/sber.joblib')


@dataclass
class Mortgage:
    name: str
    conditions: str
    url: str = ''
        

MORTGAGES = {
    1: Mortgage(name='Ипотека для IT', conditions='Первоначальный взнос от 15%, процентная ставка от 4,7%', url='https://domclick.ru/ipoteka/programs/it-workers?utm_source=sberbank&utm_medium=referral&utm_campaign=homenew&utm_referrer=https%3A%2F%2Fwww.sberbank.ru%2F'),
    2: Mortgage(name='Вторичное жильё', conditions='Первоначальный взнос от 0%, процентная ставка 10,5%',  url='https://www.sberbank.ru/ru/person/credits/home/buying_complete_house'),
    3: Mortgage(name='Новостройка', conditions='Первоначальный взнос от 0%, процентная ставка 10,5%',  url='https://www.sberbank.ru/ru/person/credits/home/buying_project'),
    4: Mortgage(name='Семейная ипотека', conditions='Первоначальный взнос от 15%, процентная ставка от 5,3%',  url='https://www.sberbank.ru/ru/person/credits/home/family'),
    5: Mortgage(name='Ипотека с господдержкой', conditions='Первоначальный взнос от 15%, специальная ставка 0,1% годовых на первый год кредита, далее ставка составит от 6,3% годовых',  url='https://www.sberbank.ru/ru/person/credits/home/gos_2020'),
    6: Mortgage(name='Ипотека по двум документам', conditions='Первоначальный взнос от 15%, процентная ставка от 5,3%',  url='https://www.sberbank.ru/ru/person/credits/home/ipoteka-po-dvum-dokumentam'),
    7: Mortgage(name='Дальневосточная ипотека', conditions='Первоначальный взнос от 15%, процентная ставка от 1,5%',  url='https://www.sberbank.ru/ru/person/credits/home/buying_complete_house_daln'),
    8: Mortgage(name='Ипотека на строительство дома', conditions='Первоначальный взнос от 25%, процентная ставка от 5%',  url='https://www.sberbank.ru/ru/person/credits/home/building'),
    9: Mortgage(name='Наличные под залог', conditions='Процентная ставка от 10,8%',  url='https://www.sberbank.ru/ru/person/credits/money/credit_zalog'),
    10: Mortgage(name='Молодёжная ипотека', conditions='Первоначальный взнос от 15%, специальная ставка 0,1% годовых на первый год кредита, далее ставка составит от 10,5% годовых',  url='https://www.sberbank.ru/ru/person/credits/home/molodezhnaya-ipoteka'),
    11: Mortgage(name='Комната в ипотеку', conditions='Первоначальный взнос от 20%, процентная ставка от 10,5%',  url='https://www.sberbank.ru/ru/person/credits/home/komnata'),
    12: Mortgage(name='Ипотека на загородную недвижимость и землю', conditions='Первоначальный взнос от 25%, процентная ставка от 10,8%',  url='https://www.sberbank.ru/ru/person/credits/home/buying_cottage'),
    13: Mortgage(name='Ипотека с материнским капиталом', conditions='Процентная ставка от 10,5%',  url='https://www.sberbank.ru/ru/person/credits/home/mot'),
    14: Mortgage(name='Ипотека для иностранных граждан', conditions='Первоначальный взнос от 20%, процентная ставка от 10,8%',  url='https://www.sberbank.ru/ru/person/credits/home/ipoteka-dlya-inostrancev'),
    15: Mortgage(name='Ипотека на гараж, машино-место или кладовую', conditions='Первоначальный взнос от 25%, процентная ставка от 11%',  url='https://www.sberbank.ru/ru/person/credits/home/garage'),
    16: Mortgage(name='Рефинансирование ипотеки', conditions='Процентная ставка 6,8%',  url='https://www.sberbank.ru/ru/person/credits/refinancing_mortgages'),
    17: Mortgage(name='Военная ипотека', conditions='Первоначальный взнос от 15%, процентная ставка от 9,9%',  url='https://www.sberbank.ru/ru/person/credits/home/mil')
}


STOP_WORDS = ['ипотека', 'ипотеку', 'ипотеке', 'ипотеки', 'в', 'на', 'по', 'под', 'при', 'с']
MORPH = pymorphy2.MorphAnalyzer()
def lemmatize(corp):    
    lem_corp = []
    for text in corp:
        lemmas = [
            MORPH.parse(it)[0].normal_form
            for it in text.lower().split()
            if it not in STOP_WORDS
        ]
        lem_corp.append(' '.join(lemmas))
    return lem_corp


@sleep_and_retry
@limits(calls=1, period=1)
def download_keywords_page(text):
    query = '+'.join(text.split())
    response = requests.get(f'https://www.bukvarix.com/keywords/?q={query}')
    response.raise_for_status()
    return BeautifulSoup(response.content, 'lxml')


REPORT_REGEX = re.compile(r"[^/]+\.csv$")
def load_similar_queries(text):
    try:
        keywords_page = download_keywords_page(text)
        file_href = keywords_page.find('a', class_='report-download-button').get('href')
        filename = REPORT_REGEX.search(file_href).group(0)
        file_path = file_href.replace(filename, '')
        filename = urllib.parse.quote_plus(urllib.parse.unquote(filename))
        result_df = pd.read_csv(f'https://www.bukvarix.com{file_path}{filename}', sep=';')
        return [text.lower()] + result_df['Ключевое слово'].values.tolist()
    except:
        logger.exception(f"Can't download similar queries for {text}")
        return [text.lower()]


def build_readmore_markup(mortgage):
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("Читать условия подробнее", url=mortgage.url)
    ]])


def start(update, context): 
    update.message.reply_text("Добрый день, информация по какому ипотечному продукту Вас интересует?")


def predict(update, context):  
    test_corpora_lem = lemmatize(load_similar_queries(update.message.text))
    
    X_test_vector = TFIDF_MODEL.transform(test_corpora_lem)
    y_pred_proba = CLF.predict_proba(X_test_vector)
    
    mean_proba = np.mean(y_pred_proba, axis = 0)
    proba_dict = {i+1: mean_proba[i] for i in range(len(mean_proba))}

    best_options = sorted(proba_dict.items(), key=lambda x: x[1], reverse=True)
    
    answer = MORTGAGES[best_options[0][0]]    
    update.message.reply_text(
        f"{answer.name}: {answer.conditions}",
        reply_markup=build_readmore_markup(answer),
    )
    
    answer_2 = MORTGAGES[best_options[1][0]]
    answer_3 = MORTGAGES[best_options[2][0]]
    
    other_options_keyboard = [
        [InlineKeyboardButton(answer_2.name, callback_data=best_options[1][0])],
        [InlineKeyboardButton(answer_3.name, callback_data=best_options[2][0])],
        [InlineKeyboardButton("Спасибо, всё верно", callback_data=END_BUTTON)],
    ]

    update.message.reply_text(
        'Возможно вы хотели получить информацию о другой услуге?',
        reply_markup=InlineKeyboardMarkup(other_options_keyboard),
    )


def button(update, context):
    query = update.callback_query
    query.answer()
    
    data = update.callback_query.data
    if data.isdigit():
        mortgage_info = MORTGAGES[int(data)]
        query.edit_message_text(
            text=f"{mortgage_info.name}: {mortgage_info.conditions}",
            reply_markup=build_readmore_markup(mortgage_info),
        )
    elif data == END_BUTTON:
        query.delete_message()
    else:
        logger.warn(f"Unknown button, {data}")


def main():
    updater = Updater(token=TOKEN, use_context=True)

    updater.dispatcher.add_handler(CommandHandler('start', start))
    updater.dispatcher.add_handler(CallbackQueryHandler(button))
    
    predict_handler = MessageHandler(Filters.text & (~Filters.command), predict)
    updater.dispatcher.add_handler(predict_handler)
    
    updater.start_polling()
    updater.idle()


if __name__ == "__main__":
	main()