import pandas as pd
import numpy as np
import scipy as sp
import nltk
import spacy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import metrics
from textblob import TextBlob, Word
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import sent_tokenize
from nltk.chunk.regexp import RegexpParser
from nltk.chunk import tree2conlltags
from nltk import word_tokenize, sent_tokenize
from gensim.models import Word2Vec
from spacy import displacy
from nltk.tokenize import word_tokenize
from scipy.spatial import distance
from nltk import pos_tag
from nltk import RegexpParser
nlp = spacy.load("en_core_web_sm")
nltk.download('punkt')
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
import warnings
warnings.filterwarnings('ignore')
from PIL import Image
plt.style.use('fivethirtyeight')
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

comments=pd.read_excel('/Users/akshayravi/Downloads/surveycomments2017.xlsx')
print(comments.head(5))

print("The columns in the file are the following-\n",comments.columns,'\n')
column_name=input("Please enter the column name.\n")

RefSentences=[]
ref_sent=''
ref_sent=input("Please enter the reference sentence.\n")
RefSentences.append(ref_sent)

top_n=input("Please enter the number of top sentences you want to return.\n")
top_n=int(top_n)

def Data_Cleaning(Unclean_df):
  Unclean_df[column_name]=Unclean_df[column_name].fillna('No Comment')
  Unclean_df['Tags']=Unclean_df['Tags'].fillna('Overall Feedback')
  Unclean_df[column_name] = Unclean_df[column_name].str.replace(r"[^a-zA-Z.?!' ]+", "")
  Unclean_df[column_name] = Unclean_df[column_name].str.replace('\n', '')
  Unclean_df[column_name] = Unclean_df[column_name].str.replace('\t', ' ')
  Unclean_df[column_name] = Unclean_df[column_name].str.replace(' {2,}', ' ', regex=True)
  Unclean_df[column_name] = Unclean_df[column_name].str.strip()
  Unclean_df[column_name] = Unclean_df[column_name].str.lower()
  Unclean_df = Unclean_df.replace({column_name: {"none":"no comment", "none.":"no comment","n/a":"no comment", "na":"no comment" ,"nil":"no comment", "-":"no comment",
                                            "none at this time.":"no comment",
                                            "none to add":"no comment",
                                            "nothing else to add.":"no comment",
                                            "there is no other comment at this time.":"no comment",
                                            "no further comments.":"no comment",
                                            "no comments at this time.":"no comment",
                                            "":"no comment"
                                            }})
  return Unclean_df

Data_Cleaning(comments)
#print(comments.head(5))

comments['tokenized_sentences']=comments[column_name].apply(sent_tokenize)
sentences=[]
for sentence in comments['tokenized_sentences']:
  for s in sentence:
    sentences.append(s)
df=pd.DataFrame(sentences,columns=[column_name])
comments=df
#print(comments.head())

#Splitting the sentences into an array of words
Responses = comments.Responses.map(lambda Responses: Responses.split())
#print(Responses.head(5))

def Sentence_Splitter(Input_df):
  Input_df['tokenized_sentences']=Input_df[column_name].apply(sent_tokenize)
  sentences=[]
  for sentence in Input_df['tokenized_sentences']:
    for s in sentence:
      sentences.append(s)
  df=pd.DataFrame(sentences,columns=[column_name])
  Input_df=df
  Responses = Input_df.Responses.map(lambda Responses: Responses.split())
  return Input_df,Responses

def Train_Model(Dataset_Name,Inp_Responses,Inp_Df):
  model = Word2Vec(workers=4,vector_size=64,min_count=2,window=7,sg=1,sample=1e-3,negative=10,alpha=0.01,min_alpha=0.00033)
  model.build_vocab(Inp_Responses, progress_per=1000)
  model.train(Inp_Df, total_examples=model.corpus_count, epochs=30, report_delay=1)
  model.init_sims(replace=True)
  model.save(f"{Dataset_Name}.model")

Train_Model("surveycomments2017",Responses,comments)

#Frequency count
word_count=pd.Series(' '.join(comments.Responses).split()).value_counts()
freq_count={}
total_word_count=0

for i in range(0,len(word_count)):
  if(word_count[i] not in freq_count):
    freq_count[word_count[i]]=1
  else:
    freq_count[word_count[i]]+=1
  total_word_count+=word_count[i]

#print("Total No of words ",total_word_count)
#print("unique word count ", len(word_count))
#print("No of words that occur 1 times ",freq_count[1])

sum=0
for idx, sentence in enumerate(comments[column_name]):
  sum+=len(sentence.split())
sum/=len(comments[column_name])
#print(sum)

def getPOS(r):
  text_tokens = word_tokenize(r)
  tokens_without_sw = [word for word in text_tokens if not word.lower() in stopwords]
  filtered_sentence = (" ").join(tokens_without_sw)
  doc = nlp(filtered_sentence)
  return doc.noun_chunks


# A function to find top n sentences similar to reference sentence based on trained word2vec model
def find_top_K_word2vec(reference, topK, Dataset_Name, Input_Df):
    Input_Df['SimilarityScore'] = None
    sentences = Input_Df[column_name]
    # Splitting the reference sentence into words after converting to lowercase
    reference_splitted = reference.lower().split()
    # Loading the word2vec pre-trained model
    model_336 = Word2Vec.load(f"{Dataset_Name}.model")
    # Adding the words from the new reference sentence to this model's vocab
    model_336.build_vocab(reference_splitted, update=True)
    # Retraining the model to separately embed the reference sentence to it
    model_336.train([reference_splitted], total_examples=1, epochs=30)
    w2v_vocab = set(model_336.wv.key_to_index)
    sentences_similarity = np.zeros(len(sentences))
    indexing = np.zeros(len(sentences))
    target_sentence_words = [w for w in reference.split() if w in w2v_vocab]
    # Iterating over the survey dataset
    for idx, sentence in enumerate(sentences):
        i=idx
        sentence_words=[w for w in sentence.split() if w in w2v_vocab]
        if len(sentence_words)>0 and len(target_sentence_words)>0:
            # Using cosine similarity based function n_similarity to find similarity score between two set of words
            sim_score = model_336.wv.n_similarity(target_sentence_words, sentence_words)
            indexing[idx] = idx
            noun_chunk_source = getPOS(Input_Df[column_name][i])
            noun_chunk_target = getPOS(reference)
            lst_src = (list(noun_chunk_source))
            lst_dest = (list(noun_chunk_target))
            for j in range(len(lst_src)):
                chunk_src = lst_src[j]
                for k in range(len(lst_dest)):
                    chunk_dest = lst_dest[k]

                    # src_encode=chunk_src.root.text.lower()
                    # dest_encode=chunk_dest.root.text.lower()

                    src_encode = chunk_src.text.lower()
                    dest_encode = chunk_dest.text.lower()

                    # extra_score=1 - distance.cosine(embed([src_encode]),embed([dest_encode]))

                    src_encode1 = [w for w in src_encode.split() if w in w2v_vocab]
                    dest_encode1 = [w for w in dest_encode.split() if w in w2v_vocab]

                    if len(src_encode1) > 0 and len(dest_encode1) > 0:
                        extra_score1 = model_336.wv.n_similarity(src_encode1, dest_encode1)

                        if (chunk_src.text.lower() == chunk_dest.text.lower()):
                            # print(Input_Df[column_name][i])
                            # print(sim_score)
                            # print("full:",chunk_src.text.lower())
                            sim_score += 5
                            # print(sim_score)
                        elif (chunk_src.root.text.lower() == chunk_dest.root.text.lower()):
                            # print(Input_Df[column_name][i])
                            # print(sim_score)
                            # print("root:",chunk_src.root.text.lower())
                            sim_score += 3 + extra_score1 * 2
                            # print(sim_score)
                        elif (
                                " " + chunk_src.root.text.lower() + " " in " " + chunk_dest.text.lower() + " " or " " + chunk_dest.root.text.lower() + " " in " " + chunk_src.text.lower() + " "):
                            # print(Input_Df[column_name][i])
                            # print(sim_score)
                            # print("subs:",chunk_src.root.text.lower(),"-",chunk_dest.text.lower(),"-",chunk_dest.root.text.lower(),"-",chunk_src.text.lower())
                            sim_score += 2 + extra_score1
                            # print(sim_score)
                        flg = True
            Input_Df['SimilarityScore'][i] = sim_score
        else:
            Input_Df['SimilarityScore'][i] = 0

    W2VTopK = Input_Df[[column_name, 'SimilarityScore']].sort_values('SimilarityScore', ascending=False).head(topK)
    W2VTopK.reset_index(inplace=True, drop=True)
    W2VTopK.index.names = ['Rank']
    return W2VTopK

from IPython.display import display

for r in RefSentences:
  print(r)
  print("------------------")
  temp_df=find_top_K_word2vec(r,top_n,"surveycomments2017",comments)
  display(temp_df)
  print("------------------")

def getPOS(r):
  text_tokens = word_tokenize(r)
  # tokens_without_sw = [word for word in text_tokens if not word.lower() in stopwords]
  # filtered_sentence = (" ").join(tokens_without_sw)
  filtered_sentence=r
  doc = nlp(filtered_sentence)
  return doc.noun_chunks

for r in RefSentences:
  dict_pos={}
  print(r)
  print("------------------")
  for chunk in getPOS(r):
      if str(chunk.root.text).lower() not in stopwords:
        dict_pos[chunk.root.dep_]=f'{chunk.text}  | {chunk.root.text}   | {chunk.root.dep_}   |{chunk.root.head.text} \n\n'
        # print(f'{chunk.text}  | {chunk.root.text}   | {chunk.root.dep_}   |{chunk.root.head.text} \n\n')
  if 'nsubj' in dict_pos:
    print(dict_pos['nsubj'])
  elif 'dobj' in dict_pos:
    print(dict_pos['dobj'])
  #elif:
    #print(dict_pos['pobj'])
  else:
    print("No good one.")
  print("------------------")

Sentiment_DF=pd.DataFrame()
for r in RefSentences:
    Sentiment_DF=pd.concat([Sentiment_DF,temp_df],axis=0)

Sentiment_DF['Sentiment']=None
Sentiment_DF['Sentiment_Score']=None

Sentiment_DF['Sentiment_Score']=Sentiment_DF[column_name].apply(lambda x: TextBlob(x).sentiment.polarity)
Sentiment_DF["Sentiment"]=np.select([Sentiment_DF["Sentiment_Score"] < 0, Sentiment_DF["Sentiment_Score"] == 0, Sentiment_DF["Sentiment_Score"] > 0], ['NEG', 'NEU', 'POS'])
display(Sentiment_DF)