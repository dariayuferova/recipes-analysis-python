import requests
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
# nltk.download('punkt') # only need to run once

# set web scraping
headers = {
  "User-Agent":
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
link = 'https://sallysbakingaddiction.com/recipe-index/page/'
n_of_pages = 24
unique_titles = set()
for page_number in range(n_of_pages):
    page = str(page_number) + '/'
    full_url = link + page
    response = requests.get(full_url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    answer = soup.find_all('a', {'rel':'entry-title-link'})
    for i in range(len(answer)):
        if answer[i].text not in unique_titles:
            unique_titles.add(answer[i].text)

# create dataframe for further tokenization of recipy titles
titles_df = pd.DataFrame(unique_titles, columns=['titles'])
titles_df['recipy_tokens'] = ''
stopwords = ['recipe', 'recipes', 'the', 'dessert', 'favorite', 'style', ' style', 'ingredient', 'with',
             'bake', 'baked', 'best', 'super', 'filled', 'easy', 'simple', 'perfect', 'stuffed', 'make',
             'way', 'day', 'homemade', 'how', 'ultimate', 'video', 'baking', 'double', 'like', 'from',
             'tips', 'had', 'slow', 'ever', 'key', 'simply', 'cup', 'and', 'for', 'cups', 'inch', 'slice',
             'fun', 'pound', 'pot', 'mini', 'guide', 'free', 'treats', 'quick', 'sheet', 'ingredients',
             'everything', 'small', 'classic', 'calorie']
for index, row in titles_df.iterrows():
    # lowercase words
    detail = row.titles.lower()
    # tokenize words
    detail = word_tokenize(detail)
    # filter for needed words only
    detail = [word for word in detail if len(word)>2 and word.find('+')==-1 and word not in stopwords]
    # replace duplicates
    replace_tokens = {'tarts': 'tart', 'pretzels': 'pretzel', 'apples': 'apple', 'bars': 'bar',
                      'pies': 'pie', 'iced': 'ice', 'spiced': 'spice', 'brownies': 'brownie', 'creme': 'cream',
                      'cookies': 'cookie', 'cakes': 'cake', 'mores': 'marshmallow', 'rolls': 'roll'}
    for key, value in replace_tokens.items():
        detail = [d.replace(key, value) for d in detail]
    # add tokens to dataframe column
    titles_df.at[index, 'recipy_tokens'] = [word for word in detail]

# create dataframe for analyzing keywords only
count_keywords = pd.DataFrame(titles_df.recipy_tokens.sum()).value_counts() \
    .rename_axis('keywords').reset_index(name='counts')
length = len(titles_df)
count_keywords['percentage'] = 100 * count_keywords.counts / length

# create a wordcloud
cloud_data = count_keywords.set_index('keywords').to_dict()['counts']
wordcloud2 = WordCloud(width=800, height=800,
                      background_color='white',
                      min_font_size=10, max_font_size=200).generate_from_frequencies(cloud_data)
# plot the wordcloud image
plt.figure()
plt.imshow(wordcloud2)
plt.axis('off')
plt.tight_layout(pad=0)

# display the total number of unique recipes analyzed:
print(len(titles_df))
# display top 10 used words:
print(count_keywords.head(10))
# display the frequency of specific word usage in recipes:
print(count_keywords[(count_keywords['keywords']=='lemon')])
# display wordcloud image:
plt.show()
