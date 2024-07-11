import os
import json
import time
import html
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

openai_client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

headers = {
    'Accept': '*/*',
    'Connection': 'keep-alive',
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.0) AppleWebKit/537.36 (KHTML,like Gecko) Chrome/70.0.3538.110 Safari/537.36',
    'Accept-Language':'en-US,en;q=0.9',
    'Cache-Control': 'max-age=0',
    'Upgrade-Insecure-Requests': '1'
}

def get_movie_plot(url):
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    summaries = soup.select('li.ipc-metadata-list__item div.ipc-html-content-inner-div div.ipc-html-content.ipc-html-content--base div.ipc-html-content-inner-div')
    plot = summaries[0].get_text().strip()
    # Remove the storyline author <span> element
    storyline_author = summaries[1].find('span')
    if storyline_author:
        storyline_author.decompose()
    storyline = summaries[1].get_text().strip()
    return plot, storyline

def scrape_imdb_top_250():
    url = 'https://www.imdb.com/chart/top/'
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    movies = []
    data = json.loads(soup.find('script', {'type':'application/ld+json'}).text)
    for item in tqdm(data['itemListElement']):
        # Need to unescape HTML characters to extract the title properly
        title = html.unescape(item['item'].get('alternateName', item['item']['name']))
        url = item['item']['url']
        short_plot, long_plot = get_movie_plot(url + 'plotsummary/')
        movies.append({'title': title, 'short_plot': short_plot, 'long_plot': long_plot })
        time.sleep(1)
    return pd.DataFrame(movies)

def embed_movies(movies_df, model='text-embedding-3-large'):
    movie_plots = (movies_df['short_plot'] + '\n' + movies_df['long_plot']).to_list()
    embed_res = openai_client.embeddings.create(input=movie_plots, model=model)
    movies_df['embedding'] = [d.embedding for d in embed_res.data]
    return movies_df


if __name__ == '__main__':
    print('Scraping IMDB Top 250 movies')
    movies_df = scrape_imdb_top_250()
    movies_df.to_csv('data/movies.csv', index=False)
    print('Embedding movies')
    movies_df = embed_movies(movies_df)
    movies_df.to_csv('data/movies_embeddings.csv', index=False)
