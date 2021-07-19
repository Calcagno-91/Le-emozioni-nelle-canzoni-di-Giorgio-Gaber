import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import streamlit as st
import plotly.express as px
import numpy as np
import time
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


gaber = pd.read_csv('data/gaber_final1.csv')
emotions = pd.read_csv('data/emotions.csv')




title_container = st.beta_container()
col1, col2 = st.beta_columns([1, 20])
image = Image.open('data/gaberimg.jpg')
with title_container:
    with col1:
        st.image(image, width=700)
    

def emotion (sentence):
   
    list_tokens= list(gaber[gaber['titolo']==sentence]['lemmas'])[0].split()
    song = extractSongScore(list_tokens)
    emotion =max(song, key=song.get)
    
    word_cloud(list(gaber[gaber['titolo']==sentence]['lemmas'])[0])
    st.markdown('**Album: **')
    album=(list(gaber[gaber['titolo']==sentence]['album']))
    album=st.text(' - '.join(album))
    print()
    st.markdown('**Testo: **')
    st.text(list(gaber[gaber['titolo']==sentence]['testo'])[0])
    print()
    st.markdown('**Emozione: **')
    result=st.markdown(f'Nella canzone **{sentence.capitalize()}** prevale un sentimento di **{emotion.capitalize()}**')
    plot([list(gaber[gaber['titolo']==sentence]['lemmas'])[0]])
    return result

def main():
    
    st.title('Le Emozioni nelle Canzoni di Giorgio Gaber')
    
    sentence= st.text_input('Digita il titolo della Canzone','la libertà')
    sentence=sentence.strip().lower()
    if sentence not in gaber['titolo'].values:
        
           st.text(f'La canzone {sentence} non è presente nel database, riprova')
    else:
        results= emotion(sentence)
    
        

def plot(sentence):
    level = st.slider("Guarda la Frequenza dei Termini: 1. Unigrammi, 2. Bigrammi, 3. Trigrammi", 1, 3)
    if level ==1:
        sentence=ngram_df(sentence,(1,1),10)
        fig = px.bar(sentence, x='Occorrenza', y='Termine',title= 'Frequenza termini nella canzone',template='plotly_white',
                     color_continuous_scale=px.colors.sequential.Redor,orientation='h',color='Occorrenza',text='Occorrenza')
        fig.update_layout(coloraxis_showscale=False)
    elif level==2:
        sentence=ngram_df(sentence,(2,2),10)
        fig = px.bar(sentence, x='Occorrenza', y='Termine',title= 'Frequenza Bigrammi nella canzone',template='plotly_white',
                     color_continuous_scale=px.colors.sequential.Redor,orientation='h',color='Occorrenza',text='Occorrenza')
        fig.update_layout(coloraxis_showscale=False)
    else:
        sentence=ngram_df(sentence,(3,3),10)
        fig = px.bar(sentence, x='Occorrenza', y='Termine',title= 'Frequenza Trigrammi nella canzone',template='plotly_white',
                     color_continuous_scale=px.colors.sequential.Redor,orientation='h',color='Occorrenza',text='Occorrenza')
        fig.update_layout(coloraxis_showscale=False)
    
    return st.plotly_chart(fig,use_container_width=True)

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

def ngram_df(corpus,nrange,n=None):
    vec = CountVectorizer(ngram_range=nrange).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    total_list=words_freq[:n]
    df=pd.DataFrame(total_list,columns=['Termine','Occorrenza']).sort_values('Occorrenza',ascending=True)
    return df
    


def word_cloud(sentence):
    wc = WordCloud(stopwords=STOPWORDS, 
                   background_color="white",colormap="viridis",
                   max_words=80, max_font_size=256,
                   random_state=42,width=2000,height=600,
                   )

    fig, (ax2) = plt.subplots(1,1,figsize=[10, 8])

    wc.generate(sentence)
    plt.imshow(wc.recolor(colormap='Reds'), interpolation="bilinear")
    plt.axis('off')
    return st.pyplot(fig)
# =============================================================================
# def plot2(sentence):
#     
#     plt.figure(figsize=(9,10))
#     sns.barplot(sentence["Frequenza"],sentence["Termine"])
#     plt.title("Top 10 Bigrams")
#     
#     plt.show()
# 
#     return plt.show() 
# =============================================================================

# =============================================================================
# def plot(sentence):
#     level = st.slider("Guarda la Frequenza dei Termini: 1. Unigrammi, 2. Bigrammi, 3. Trigrammi", 1, 3)
#     if level ==1:
#         sentence=ngram_df(sentence,(1,1),10)
#         fig, (ax2) = plt.subplots(1,1,figsize=[4, 4])
#         plt.figure(figsize=(9,10))
#         sns.barplot(sentence["Occorrenza"],sentence["Termine"], color ='red')
#         plt.title("Top 10 Unigrammi")
#     elif level==2:
#         sentence=ngram_df(sentence,(2,2),10)
#         fig, (ax2) = plt.subplots(1,1,figsize=[8, 6])
#         plt.figure(figsize=(9,10))
#         sns.barplot(sentence["Occorrenza"],sentence["Termine"], color ='red')
#         plt.title("Top 10 Bigrammi")
#     else:
#         sentence=ngram_df(sentence,(3,3),10)
#         fig, (ax2) = plt.subplots(1,1,figsize=[8, 6])
#         plt.figure(figsize=(9,10))
#         sns.barplot(sentence["Occorrenza"],sentence["Termine"], color ='red')
#         plt.title("Top 10 Trigrammi")
#     
#   
#     
#     return st.pyplot(plt,use_container_width=True)
# 
# =============================================================================
def extractWordScore(token):
    wordScore = {}
        
    wordEmotions = emotions[emotions['lemma'] == token]
    if wordEmotions.empty:
        wordEmotions = emotions[emotions['word'] == token]
    if not wordEmotions.empty:
        wordScore = wordEmotions.drop(['word','cosine','lemma'], axis=1).set_index('emotion').T.to_dict('record')[0]
    return wordScore

def extractSongScore(song):
    songScore = {}
    i = 0
    wordsScore = []
    for t in song:
        score = extractWordScore(t)
        if score:
            increment = False
            for k in score.keys():
                songScore[k] = songScore.get(k, 0.0) + score[k]
                if not increment:
                    i += 1
                    increment = True
                songScore[k] = songScore.get(k, 0.0) + score[k]
    if i > 0:
        for k in songScore:
            songScore[k] = songScore[k] / i
    
    return songScore

if __name__ == '__main__':
	main()



