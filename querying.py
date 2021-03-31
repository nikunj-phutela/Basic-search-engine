#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import glob
import pandas as pd
import re
import string
import itertools
import nltk
import time
from nltk.stem import PorterStemmer
stemming = PorterStemmer()
from nltk.corpus import stopwords
stops = set(stopwords.words("english")) 
#print(stops)
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
import pickle
from functools import reduce
from itertools import product
from collections import defaultdict
import json
ps =nltk.PorterStemmer()
from nltk.corpus import stopwords
stops = set(stopwords.words("english")) 
#print(stops)
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer


# In[2]:


with open('inverted_index.json') as f:
    inverted_index = json.load(f)


# In[3]:


def clean_text(text):
    str1 = " "
    text_lc = "".join([word.lower() for word in text if word not in string.punctuation]) # remove puntuation
    text_rc = re.sub('[0-9]+', '', text_lc)
    tokens = re.split('\W+', text_rc)    # tokenization
    text = [ps.stem(word) for word in tokens if word not in stops]  # remove stopwords and stemming
    return str1.join(text)


# In[4]:


def one_word_query_dict(word, invertedIndex,result):
    row_id = []
#     res = []
    final_list = []    
    word = clean_text(word)
    pattern = re.compile('[\W_]+')
    word = pattern.sub(' ',word)
    if word in invertedIndex.keys():
        l = [filename for filename in invertedIndex[word].keys()]
        for filename in l:
            final_list.append(set((invertedIndex[word][filename].keys())))
        
        for i in range(len(l)):
            if l[i] not in result:
                result[l[i]] = final_list[i]
            else:
                result[l[i]] = result[l[i]].union(final_list[i])
                    
        
    


# In[5]:


def free_text_query(string,invertedIndex):
    string = clean_text(string)
    pattern = re.compile('[\W_]+')
    string = pattern.sub(' ',string)
    result = {}
    for word in string.split():
        one_word_query_dict(word,invertedIndex,result)
    return result


# In[6]:


def phrase_query_correct(string,inverted_index):
    string = clean_text(string)
    final_dict = {}
    pattern = re.compile('[\W_]+')
    string = pattern.sub(' ',string)
    listOfDict = []
    for word in string.split():
        result = {}
        one_word_query_dict(word,inverted_index,result)
        listOfDict.append(result.copy())
    common_docs = set.intersection(*map(set, listOfDict))
    words_list = string.split()
    final_res = {}
    for filename in common_docs:
        ts = []
        for word in string.split():
               ts.append(inverted_index[word][filename])
        
        for word_pos_dict_no in range(0,len(ts)):
            for row_number in ts[word_pos_dict_no]:
                for positions in range(0,len(ts[word_pos_dict_no][row_number])):
                    ts[word_pos_dict_no][row_number][positions] -= word_pos_dict_no
        common_rows = set.intersection(*map(set,ts))
        for row_number in common_rows:
            final_list_of_pos = []
            for word_no in range(0,len(ts)):
                final_list_of_pos.append(ts[word_no][row_number])                    
            res = list(reduce(lambda i, j: i & j, (set(x) for x in final_list_of_pos)))
            if(len(res)>0):
                if(filename not in final_res):
                    final_res[filename] = []
                final_res[filename].append(row_number)
    return final_res
            
    


# In[7]:


def printresult(res,query): #prints the snippets
        docs={}
        if type(res) == type({}):
            for document,rows in res.items():
                infile = f'Dataset/{document}'
                for row in rows:
                    data = pd.read_csv(infile, skiprows = int(row) , nrows=1, usecols=[6])
                    #print({'doc_name': document, 'Snippet': data.values[0][0]})
                    #docs.append("Document name: "+document+" Snippet: "+data.values[0][0])
                    docs[data.values[0][0]]=document
                    #print()
        elif type(res) == type([]):
            for result in res:
                for document,rows in result.items():
                    infile = f'Dataset/{document}'
                    for row in rows:
                        data = pd.read_csv(infile, skiprows = int(row) , nrows=1, usecols=[6])
                        #print({'doc_name': document, 'Snippet': data.values[0][0]})
                        #docs.append("Document name: "+document+" Snippet: "+data.values[0][0])
                        docs[data.values[0][0]]=document
                        #print()
        else:
            pass
        ranking(docs,query)
   
    


# In[57]:


#printresult(phrase_query_correct("climate change",inverted_index))


# In[8]:


from rank_bm25 import BM25Okapi
def ranking(corpus,query):
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split(" ")
    doc_scores = bm25.get_scores(tokenized_query)
    #bm25.get_top_n(tokenized_query, corpus, n=1)
    #creating a dictionary with the scores
    score_dict = dict(zip(corpus, doc_scores))
    #creating list of ranked documents high to low
    #print(score_dict)
    doc_ranking = sorted(score_dict, key=score_dict.get, reverse = True)
    #get top 100
    doc_ranking = doc_ranking[0:100]
    #print(doc_ranking)
    for i in doc_ranking:
        print("score:", score_dict[i])
        print("Doc name: ",corpus[i])
        print(i)
        print()
        


# In[116]:


def enter_query():
    print("Please input the type of Query : \n '1' for free text queries \n '2' for phrase queries ")
    query_type = input()
    print("Please enter the query")
    query = input()
    start = time.time()
    try:
    	if(query_type == "1" and len(query) == 1):
        	printresult(one_word_query_dict(query,inverted_index),query)
    	elif(query_type == "1"):
        	printresult(free_text_query(query,inverted_index),query)
    	elif(query_type == "2" ):
        	printresult(phrase_query_correct(query,inverted_index),query)
    	elif(query_type != "1" and query_type != "2" ):
        	print("Please enter a valid query type")
    	else:
        	print("There are no matches for this query")
    except:
    	print('Query not in corpus')
    end = time.time()
    print("time taken:",end-start)


# In[122]:


enter_query()


# In[ ]:


enter_query()

