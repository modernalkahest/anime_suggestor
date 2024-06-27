"""
# Welcome to Streamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:.
If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""

import pandas as pd
import numpy as np
from warnings import filterwarnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from time import time
import streamlit as st

filterwarnings('ignore')
#loading the data set
anime = pd.read_csv('Anime.csv')
Ratings = anime[['Name','Type','Studio','Rating','Tags']]

#Finding null values and replacing them
Ratings['Studio']=Ratings['Studio'].fillna('Unknown')
Ratings['Tags']=Ratings['Tags'].fillna('Unknown')

#Encoding the categorical variable Studio
encoder = dict([(j,i) for i,j in enumerate(Ratings['Studio'].value_counts().index)])
Ratings.set_index('Name',inplace=True)
Ratings['Studio'] = Ratings.apply(lambda row: encoder[row['Studio']],axis=1)

#Encoding the categorical variable Type
Type_encoder = dict([(j,i) for i,j in enumerate(Ratings['Type'].unique())])
Ratings['Type'] = Ratings.apply(lambda row: Type_encoder[row['Type']],axis=1)


#Taking user input
anime_watched = st.text_input(label='What was the name of the latest anime you watched? ')

start = time()

#finding closest matches to user input
def cosine_sim(str1, str2):
    str1 = str1.lower()
    str2 = str2.lower()
    vectorizer = TfidfVectorizer().fit_transform([str1, str2])
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors)[0, 1]

def tag_sim(str1,str2):
    str1 = str1.lower()
    str2 = str2.lower()
    similarity = 0
    for i in str1.split():
        if i in str2.split():
            similarity+=1


    return similarity/len(str1.split())

end1 = time()
if anime_watched not in Ratings.index:
    #print(f'\nLooks like the Database does not have anime with this exact title: {anime_watched}.This list shows the closest matches.')
    matches = anime['Name'].apply(lambda x: cosine_sim(anime_watched,x))
    matches.index = anime['Name']
    matches = matches.sort_values(ascending=False)
    matches = matches.to_frame()
    match_list = list(enumerate(matches.head().index))
    match_list_dict = dict(match_list)
    #choice = int(input('\nChoose a number from 0 to 4 to confirm which anime you meant:'))
    anime_watched =match_list_dict[0]

st.text('How would you like your recommendations: \n 1. Similar rated and by a similar Studio \n 2. Similar Genre \n 3. Both \n')
choice = int(st.text_input(label=' Enter the serial number of your choice '))

Rate_Studio_threshold = 0
Genre_threshold = 0

if choice == 1:
    Rate_Studio_threshold = float(st.text_input(label='How similar do you want the ratings and studio to be? Enter a number between 0 to 100:'))
    Rate_Studio_threshold/=100.0

elif choice == 2:
    Genre_threshold = float(st.text_input(label='How similar do you want the genre to be? Enter a number between 0 to 100:'))
    Genre_threshold/=100.0

elif choice == 3:
    Rate_Studio_threshold = float(st.text_input(label='How similar do you want the ratings and studio to be? Enter a number between 0 to 100:'))
    Genre_threshold = float(st.text_input(label='How similar do you want the genre to be? Enter a number between 0 to 100:'))
    Rate_Studio_threshold/=100.0
    Genre_threshold/=100.0

start1=time()
#Isolating the features on which recommendations should be given
Anime_SR = Ratings[['Studio','Rating']]

#Finding similarities between user input title and existing database
Cos_Similarity = Anime_SR.apply(lambda row: np.dot(Anime_SR.loc[anime_watched],row)/(np.linalg.norm(Anime_SR.loc[anime_watched])*np.linalg.norm(row)),axis=1)

#Converting to Dataframe
Cos = Cos_Similarity.to_frame()

Anime_Tags = Ratings['Tags']

#Adding columns to the dataframe
Cos.columns = ['Cosine Similarity']
Cos['Tag Similarity'] = Anime_Tags.apply(lambda row: tag_sim(' '.join(Anime_Tags.loc[anime_watched].split(',')),' '.join(row.split(','))))

#Sorting recommendations by Cosine similarity

if choice == 3:
    Recommendation = Cos[(Cos['Cosine Similarity']>Rate_Studio_threshold)&(Cos['Tag Similarity']>Genre_threshold)].sort_values(by='Tag Similarity',ascending=False)

elif choice == 1:
    Recommendation = Cos[(Cos['Cosine Similarity']>Rate_Studio_threshold)].sort_values(by='Tag Similarity',ascending=False)

elif choice == 2:
    Recommendation = Cos[(Cos['Tag Similarity']>Genre_threshold)].sort_values(by='Tag Similarity',ascending=False)

#Getting top 5 recommendations from the entire dataframe
recommended_n = Recommendation

recommendation_list = list(recommended_n.index)
if anime_watched in recommendation_list:
    recommendation_list.remove(anime_watched)

recommendations = []

for i in recommendation_list:
    if cosine_sim(anime_watched,i)>0:
        continue

    else:
        recommendations.append(i)

if len(recommendations) == 0:
    st.text('looks like for the chosen settings there are no similar anime')

else:
    st.text('\n\nRecommended to watch next:\n\n')
    
    if len(recommendations) >=10:
        for i,j in enumerate(recommendations):
            Similarity_Score = Recommendation.loc[j]['Tag Similarity']
            st.text(f'{i+1}. {j} \n{round(Similarity_Score*100)}% similar to {anime_watched}\n\n')
            if i+1 >=10:
                break


    else:
        for i,j in enumerate(recommendations):
            Similarity_Score = Recommendation.loc[j]['Tag Similarity']
            st.text(f'{i+1}. {j} \n{round(Similarity_Score*100)}% similar to {anime_watched}\n\n')

end = time()

st.text(f'Execution Time: {round((end1-start)+(end-start1),2)} seconds')
