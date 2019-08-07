import numpy as np
import io
import pandas as pd
from stop_words import get_stop_words
from nltk.tokenize import RegexpTokenizer
from gensim import corpora, models
import gensim
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image
import PIL.ImageOps
import random
from wordcloud import ImageColorGenerator
import pdb


titles = pd.read_table("titles.txt", header = None)
titles = ['actualize', 'app_academy_test', 'general_assembly']

doc_set = [[] for x in range(len(titles))]

print(doc_set)

i = 0
for title in titles:
    song_file_name = title + '.txt'
    song_file = io.open(song_file_name,"r",encoding='utf8')
    doc_set[i] = song_file.read()
    song_file.close()
    i += 1




# pdb.set_trace()
# doc_set is the entire corpous

# Listed of stop-words found at
# https://pypi.python.org/pypi/stop-words
# I also added names of former presidents

stop_words = get_stop_words('en')
stop_words.append('actualize')
stop_words.append('GA')
stop_words.append('general')
stop_words.append('assembly')
stop_words.append('App')
stop_words.append('Academy')
stop_words.append('academy')
# stop_words.extend(["washington","adams","jefferson","madison","monroe","adams","jackson","van","buren",
#                      "harrison","tyler","polk","taylor","fillmore","pierce","buchanan","lincoln","johnson",
#                      "grant","hayes","garfield","arthur","cleveland","harrison","mckinley","roosevelt","taft",
#                      "wilson","harding","coolidge","hoover","truman","eisenhower","kennedy","nixon","ford",
#                      "carter","reagan","bush","clinton","obama","michelle","roberts"])

# Clean up text
tokenizer = RegexpTokenizer(r'\w+')
texts = []
for i in doc_set:
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)
    stopped_tokens = [i for i in tokens if not i in stop_words]
    longer_tokens = [i for i in stopped_tokens if len(i) > 2]
    texts.append(longer_tokens)

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# Run tf-idf
tfidf = models.TfidfModel(corpus)

## TRUMP

# Based off example:
# https://github.com/amueller/word_cloud/blob/master/examples/a_new_hope.py

def orange_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return "hsl(24, 99%%, %d%%)" % random.randint(40, 70)

# Trump mask found at
# https://img1.etsystatic.com/140/0/6522319/il_fullxfull.990448319_izew.jp


# trump_mask = np.array(Image.open("trump_mask.jpg"))
top_words = np.sort(np.array(tfidf[corpus[0]],dtype = [('word',int), ('score',float)]),order='score')[::-1]
list_of_words = {}
for word,score in top_words:
    list_of_words[dictionary[word]] = score

# pdb.set_trace()
# wc = WordCloud(background_color="white").fit_words([(dictionary[word],score) for word,score in top_words])
wc = WordCloud(background_color="white").fit_words(list_of_words)

plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
wc.to_file("actualize_wordcloud.png")

## OBAMA

# Based off example:
# https://github.com/amueller/word_cloud/blob/master/examples/colored.py
# Obama mask found at
# https://tmillan1.files.wordpress.com/2012/03/nobackgroundobama.png

# obama_mask = np.array(Image.open("obama_mask.png"))
# top_words = np.sort(np.array(tfidf[corpus[55]],dtype = [('word',int), ('score',float)]),order='score')[::-1]
# wc = WordCloud(background_color="white", max_words=2000, mask=obama_mask,random_state=42)
# wc.fit_words([(dictionary[word],score) for word,score in top_words])
# image_colors = ImageColorGenerator(obama_mask)

# plt.imshow(wc.recolor(color_func=image_colors))
# plt.axis("off")
# wc.to_file("obama_wordcloud_2009.png")

## BUSH

# Bush mask found at
# https://lhhs.neocities.org/georgebush.png

# bush_mask = np.array(Image.open("bush_mask.png"))
# top_words = np.sort(np.array(tfidf[corpus[54]],dtype = [('word',int), ('score',float)]),order='score')[::-1]
# wc = WordCloud(background_color="white", max_words=2000, mask=bush_mask,
#                random_state=42)
# wc.fit_words([(dictionary[word],score) for word,score in top_words])
# image_colors = ImageColorGenerator(bush_mask)

# plt.imshow(wc.recolor(color_func=image_colors))
# plt.axis("off")
# wc.to_file("bush_wordcloud_2005.png")
