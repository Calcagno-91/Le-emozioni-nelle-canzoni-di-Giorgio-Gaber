import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import streamlit as st
import plotly.express as px
import numpy as np
import time
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import spacy
import spacy.cli
spacy.cli.download('it_core_news_sm')
nlp = spacy.load('it_core_news_sm')

gaber = pd.read_csv('data/gaber_final1.csv')
emotions = pd.read_csv('data/emotions.csv')




title_container = st.beta_container()
image = Image.open('data/gaberimg.jpg')

st.image(image)

# =============================================================================
# @inproceedings{DBLP:conf/lrec/PassaroL16,
#   author    = {Lucia C. Passaro and
#                Alessandro Lenci},
#   editor    = {Nicoletta Calzolari and
#                Khalid Choukri and
#                Thierry Declerck and
#                Sara Goggi and
#                Marko Grobelnik and
#                Bente Maegaard and
#                Joseph Mariani and
#                H{\'{e}}l{\`{e}}ne Mazo and
#                Asunci{\'{o}}n Moreno and
#                Jan Odijk and
#                Stelios Piperidis},
#   title     = {Evaluating Context Selection Strategies to Build Emotive Vector Space
#                Models},
#   booktitle = {Proceedings of the Tenth International Conference on Language Resources
#                and Evaluation {LREC} 2016, Portoro{\v{z}}, Slovenia, May 23-28, 2016},
#   publisher = {European Language Resources Association {(ELRA)}},
#   year      = {2016},
#   url       = {http://www.lrec-conf.org/proceedings/lrec2016/summaries/637.html},
#   timestamp = {Mon, 19 Aug 2019 15:22:52 +0200},
#   biburl    = {https://dblp.org/rec/conf/lrec/PassaroL16.bib},
#   bibsource = {dblp computer science bibliography, https://dblp.org}
# }
# 
# 
# =============================================================================


def emotion (sentence):
   
    song = extractWordScore(list(gaber[gaber['titolo']==sentence]['lemmas'])[0])
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
    radar_chart(song)

    plot([list(gaber[gaber['titolo']==sentence]['lemmas'])[0]])

    return result

def main():
    html_temp = """<div style="background-color:#C72A2A;padding:1.0px;border-radius: 25px ;border-radius: 25px;opacity: 0.7">
    <h1 style="color:white;text-align:center;">Le Emozioni nelle Canzoni di Giorgio Gaber</h1>
    </div><br>"""
    st.markdown(html_temp,unsafe_allow_html=True)    
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
        fig = px.bar(sentence, x='Occorrenza', y='Termine',template='plotly_white',
                     color_continuous_scale=px.colors.sequential.Reds,orientation='h',color='Occorrenza',text='Occorrenza')
        fig.update_layout(coloraxis_showscale=False, plot_bgcolor='rgba(0,0,0,0)')
    elif level==2:
        sentence=ngram_df(sentence,(2,2),10)
        fig = px.bar(sentence, x='Occorrenza', y='Termine',template='plotly_white',
                     color_continuous_scale=px.colors.sequential.Reds,orientation='h',color='Occorrenza',text='Occorrenza')
        fig.update_layout(coloraxis_showscale=False,plot_bgcolor='rgba(0,0,0,0)')
    else:
        sentence=ngram_df(sentence,(3,3),10)
        fig = px.bar(sentence, x='Occorrenza', y='Termine',template='plotly_white',
                     color_continuous_scale=px.colors.sequential.Reds,orientation='h',color='Occorrenza',text='Occorrenza')
        fig.update_layout(coloraxis_showscale=False,plot_bgcolor='rgba(0,0,0,0)')
    for data in fig.data:
	data["width"]=0.69
	
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
    return st.pyplot(fig,use_container_width=True)



def radar_chart (song):
    fig = px.line_polar(r=list(song.values()),theta=list(song.keys()),line_close=True,template='plotly_white')



    fig.update_traces(fill='toself',line_color = 'red',mode="markers+lines")
    return st.plotly_chart(fig,use_container_width=True)




def extractWordScore(token):
    wordScore = {}
    songScore = {}
    token= nlp(token)
    taggedtokens = [token.pos_ for token in token]
    list_token=list(zip(token,taggedtokens)) 
    for i in list_token:
        wordEmotions = emotions[(emotions['lemma']==str(i[0]))&(emotions['pos']==str(i[1]))]
        if not wordEmotions.empty:
            wordScore = wordEmotions.drop(['word','cosine','lemma','pos'], axis=1).set_index('emotion').T.to_dict('record')[0]
            for k in wordScore.keys():
                songScore[k] = songScore.get(k, 0.0) + wordScore[k]
        
    return songScore





if __name__ == '__main__':
	main()



