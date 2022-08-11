 # Importing essential libraries
from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
import re
from sklearn.metrics.pairwise import sigmoid_kernel
import pandas as pd




# Load the LogisticRegression model


app = Flask(__name__)



df_2=pd.read_csv("titles.csv",usecols=['id', 'title', 'description' , 'genres' ,'imdb_score' ,'imdb_votes' , 'tmdb_popularity' ,'tmdb_score'],
    dtype={'id': 'str', 'title': 'str', 'description': 'str','genres': 'str'})

df_2 = df_2.dropna()

genres = df_2["genres"]

gene = []

for i in genres:
    string_without_brackets = re.sub(r"[\[\]\'\'\,]",'',i)
    gene.append(string_without_brackets)
    
df_2["genres"] = gene    
    
df_2["Overall description"] = df_2['description'] +" "+ df_2["genres"]
df_2

df = df_2[df_2["imdb_votes"]>1000]    
    
df = df.reset_index()


from statistics import harmonic_mean
df['overall_rating'] = df[['imdb_score','imdb_votes']].apply(lambda row : harmonic_mean(row), axis=1)


from sklearn.feature_extraction.text import TfidfVectorizer


tfv = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3),
            stop_words = 'english')
    


# Filling NaNs with empty string
df['Overall description'] = df['Overall description'].fillna('')



# Fitting the TF-IDF on the 'overview' text
tfv_matrix = tfv.fit_transform(df['Overall description'])



# Compute the sigmoid kernel
sig = sigmoid_kernel(tfv_matrix, tfv_matrix)

# Reverse mapping of indices and movie titles
indices = pd.Series(df.index, index=df['title']).drop_duplicates()




def give_rec(title, sig=sig):
    # Get the index corresponding to original_title
    idx = indices[title]

    # Get the pairwsie similarity scores 
    sig_scores = list(enumerate(sig[idx]))

    # Sort the movies 
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)

    # Scores of the 10 most similar movies
    sig_scores = sig_scores[1:6]

    # Movie indices
    course_indices = [i[0] for i in sig_scores]

    # Top 10 most similar movies
    output = df.iloc[course_indices].sort_values(by=['overall_rating','tmdb_popularity'], ascending=False)
    #return df.iloc[course_indices].sort_values(by=['overall_rating','tmdb_popularity'], ascending=False)
    return output["title"]


@app.route('/')
def home():
    return render_template('main.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':

        broswer = request.form.get('browser')

        output = list(give_rec(broswer))
        

        
        return render_template('result.html', prediction=output)
        

if __name__ == '__main__':
     app.run(debug=True)
