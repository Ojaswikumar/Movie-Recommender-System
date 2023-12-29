# MOVIE-RECOMMENDER

### Movie Recommendation System Using Cosine Similarity

Utilizing historical user interactions, MOVIE-RECOMMENDER suggests films to users by assessing their similarities. It proposes movies with a recommendation rate surpassing the user's preference rate for similar films. Essentially, it provides recommendations that diverge from others' preferences but may appeal to the individual user.


## Table of Content
- [Introduction to Recommendation System](#introduction-to-recommendation-system)
- [Cosine Similarity](#cosine-similarity)
- [Code](#code)

#### Introduction to Recommendation System
Recommendation systems are crafted to suggest items to users, drawing on a variety of factors. These systems forecast items that users are inclined to buy or find interesting. Major corporations like Google, Amazon, and Netflix employ recommendation systems to assist users in making product or movie choices. These systems operate through Content-Based Filtering, which recommends items based on past user activities, and Collaborative-Based Filtering, which suggests items preferred by users with similar tastes.

#### Cosine Similarity 
Cosine similarity is a metric used to measure how similar two items are. Mathematically it calculates the cosine of the angle between two vectors projected in a multidimensional space. Cosine similarity is advantageous when two similar documents are far apart by Euclidean distance(size of documents) chances are they may be oriented closed together. The smaller the angle, higher the cosine similarity.
```
1 - cosine-similarity = cosine-distance
```

![cosine-sim](https://github.com/garooda/Movie-Recommendation-Sysetm/blob/main/images/cosine%20sim%20%201.PNG)

![cos-form](https://bit.ly/33baNhZ)


##### Importing the important libraries

```python3
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import nltk
from nltk.stem.porter import PorterStemmer
```
##### Loading the dataset and converting it into dataframe using pandas

```python3
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')
```

##### Merge the DataFrames
merging on the basis of title
```python3
movies = movies.merge(credits,on='title')
```
##### Useful Feature
```python3
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
```

##### Dropping Null Values
```python3
movies.dropna(inplace = True)
```
##### Converting genre and keyword feature values
```python3
def convert(obj):
  L=[]
  for i in ast.literal_eval(obj):
    L.append(i['name'])
  return L
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
```

##### Converting cast  values
```python3
def convert3(obj):
  L=[]
  counter = 0
  for i in ast.literal_eval(obj):
    if counter != 3:
        L.append(i['name'])
        counter+=1
    else:
         break
  return L
movies['cast'] = movies['cast'].apply(convert3)
```

##### Converting  crew  values
```python3
def fetch_director(obj):
  L=[]
  for i in ast.literal_eval(obj):
    if i['job'] == 'Director':
     L.append(i['name'])
     break
  return L
movies['crew'] = movies['crew'].apply(fetch_director)
```

##### Splitting Overview values
```python3
movies['overview'] = movies['overview'].apply( lambda x:x.split())
movies['genres'] = movies['genres'].apply( lambda x:[i.replace(" ","") for i in x] )
movies['keywords'] = movies['keywords'].apply( lambda x:[i.replace(" ","") for i in x] )
movies['cast'] = movies['cast'].apply( lambda x:[i.replace(" ","") for i in x] )
movies['crew'] = movies['crew'].apply( lambda x:[i.replace(" ","") for i in x] )
```

##### Combining the attributes to form a tag 
```python3
movies['tags'] = movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']
```
##### Creating a new data frame with tag and doing preprocessing
```python3
new_df = movies[['movie_id','title','tags']]
new_df['tags'] = new_df['tags'].apply( lambda x:" ".join(x))
new_df['tags'] = new_df['tags'].apply( lambda x:x.lower())
```


##### Using Count-Vectorizer
```python3
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words='english')
vectors=cv.fit_transform(new_df['tags']).toarray()
```

##### Cosine-Distance
```python3
similarity=cosine_similarity(vectors)
```
##### Predicting
```python3
def recommend(movie):
    movie_index=new_df[new_df['title']==movie].index[0]
    distances=similarity[movie_index]
    movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
recommend('Avatar')
```


















