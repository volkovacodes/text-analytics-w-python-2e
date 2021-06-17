#%%
###########################################
######## STEP 1. Clean html ###############
###########################################


### how to clean texts
import requests

data = requests.get('http://www.gutenberg.org/cache/epub/8001/pg8001.html')
content = data.content

### drop html tags
import re
from bs4 import BeautifulSoup

def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    [s.extract() for s in soup(['iframe', 'script'])]
    stripped_text = soup.get_text()
    stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
    return stripped_text

clean_content = strip_html_tags(content)

###########################################
#### STEP 2. Tokenize Sentences ###########
###########################################

import nltk
#nltk.download('gutenberg')
#nltk.download('punkt')
#nltk.download('wordnet')
from nltk.corpus import gutenberg
from pprint import pprint
import numpy as np

alice = gutenberg.raw(fileids='carroll-alice.txt')
default_st = nltk.sent_tokenize
alice_sentences = default_st(text=alice)


###########################################
###### STEP 3. Tokenize Words #############
###########################################

default_wt = nltk.word_tokenize
words = default_wt(alice)
np.array(words)

### find index positions for words 
TOKEN_PATTERN = r'\w+'        
regex_wt = nltk.RegexpTokenizer(pattern=TOKEN_PATTERN,
                                gaps=False)
word_indices = list(regex_wt.span_tokenize(alice))


###########################################
###### STEP 4. Unicode and Rule ###########
###########################################

import unicodedata

def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

print(remove_accented_chars('Sómě Áccěntěd těxt'))



###########################################
###### STEP 5. Correct spelling ###########
###########################################
import contractions 
print(contractions.fix("Y'all can't expand contractions I'd think"))

###########################################
###### STEP 6. Remove Special Chars #######
###########################################

def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text

print(remove_special_characters("Well this was fun! What do you think? 123#@!", 
                          remove_digits=True))



###########################################
######## STEP 7. Correct Spelling #########
###########################################

from textblob import Word

w = Word('fianlly')
print(w.correct())

w = Word('flaot')
print(w.spellcheck())


###########################################
############ STEP 8. Stemming #############
###########################################

# there are several stemmers: PorterStemmer, LacasterStemmer etc
# I will go with SS because I know it 
from nltk.stem import SnowballStemmer
ss = SnowballStemmer("english")
print(ss.stem("jumping"))


###########################################
######## STEP 9. Lemmatization ############
###########################################
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
print(wnl.lemmatize('running', 'v'))
print(wnl.lemmatize('ate', 'v'))