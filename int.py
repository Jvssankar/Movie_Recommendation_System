import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
from nltk.stem import PorterStemmer
import numpy as np
from scipy.sparse import hstack

movies = pd.read_csv('data/tmdb_5000_movies.csv')
credits = pd.read_csv('data/tmdb_5000_credits.csv')

movies = movies.merge(credits, on='title')
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

movies.dropna(inplace=True)
movies.drop_duplicates(inplace=True)

def convert(text):
    return [i['name'] for i in ast.literal_eval(text)]

def convert_cast(text):
    return [i['name'] for i in ast.literal_eval(text)[:3]]

def fetch_director(text):
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            return [i['name']]
    return []

def remove_space(L):
    return [i.replace(" ", "") for i in L]

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert_cast)
movies['crew'] = movies['crew'].apply(fetch_director)
movies['overview'] = movies['overview'].apply(lambda x: x.split())
movies['cast'] = movies['cast'].apply(remove_space)
movies['crew'] = movies['crew'].apply(remove_space)
movies['genres'] = movies['genres'].apply(remove_space)
movies['keywords'] = movies['keywords'].apply(remove_space)

ps = PorterStemmer()
def stems(text):
    return " ".join([ps.stem(i) for i in text.split()])

def vectorize_feature(feature_column):
    cv = CountVectorizer(max_features=5000, stop_words='english')
    return cv.fit_transform(feature_column.apply(lambda x: " ".join(x)).apply(stems))

features = ['genres', 'keywords', 'cast', 'crew', 'overview']
selected_features = []
remaining_features = features.copy()
best_similarity = -1

while remaining_features:
    best_feature = None
    for feature in remaining_features:
        temp_features = selected_features + [feature]
        combined_vector = hstack([vectorize_feature(movies[f]) for f in temp_features])
        current_similarity = cosine_similarity(combined_vector).mean()
        if current_similarity > best_similarity:
            best_similarity = current_similarity
            best_feature = feature
    if best_feature:
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)
    else:
        break

combined_vector = hstack([vectorize_feature(movies[f]) for f in selected_features])
final_similarity = cosine_similarity(combined_vector)

new_df = movies[['movie_id', 'title']].copy()
new_df['tags'] = movies[selected_features].apply(lambda x: " ".join([" ".join(i) for i in x]), axis=1)
new_df['tags'] = new_df['tags'].apply(stems)

def recommend_content_based(movie, k=5):
    if movie not in new_df['title'].values:
        print(f"Movie '{movie}' not found in the dataset.")
        return
    
    index = new_df[new_df['title'] == movie].index[0]
    distances = sorted(list(enumerate(final_similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movies = [new_df.iloc[i[0]].title for i in distances[1:k+1]]
    
    relevance = [1] * len(recommended_movies)
    print(f"\nContent-Based Recommendations for '{movie}': {recommended_movies}")
    evaluate_metrics(relevance, k)

def evaluate_metrics(relevance, k):
    retrieved = len(relevance)
    precision = sum(relevance[:k]) / retrieved if retrieved > 0 else 0
    recall = sum(relevance[:k]) / sum(relevance) if sum(relevance) > 0 else 0
    dcg = sum([rel / np.log2(idx + 2) for idx, rel in enumerate(relevance[:k])])
    idcg = sum([1 / np.log2(idx + 2) for idx in range(min(k, sum(relevance)))] or [1])
    ndcg = dcg / idcg if idcg > 0 else 0

    print(f"\nMetrics for K={k}:")
    print(f"Precision@{k}: {precision:.2f}")
    print(f"Recall@{k}: {recall:.2f}")
    print(f"nDCG@{k}: {ndcg:.2f}")

ratings_data = {
    'user_id': [1, 1, 2, 2, 3, 3, 4, 4],
    'movie_id': [1, 2, 2, 3, 1, 4, 3, 4],
    'rating': [5, 4, 3, 2, 5, 4, 3, 5]
}
ratings = pd.DataFrame(ratings_data)

user_item_matrix = ratings.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)

item_similarity = cosine_similarity(user_item_matrix.T)
item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

def recommend_collaborative(movie_id, user_id, k=5):
    if movie_id not in item_similarity_df.index:
        print(f"Movie ID '{movie_id}' not found in the dataset.")
        return
    
    user_ratings = user_item_matrix.loc[user_id]
    similar_items = item_similarity_df[movie_id]
    recommendations = similar_items.sort_values(ascending=False).index[:k+1]
    recommendations = [rec for rec in recommendations if rec != movie_id][:k]
    
    relevance = [1] * len(recommendations)
    print(f"\nCollaborative Recommendations for Movie ID '{movie_id}' and User ID '{user_id}': {recommendations}")
    evaluate_metrics(relevance, k)

recommend_content_based("The Dark Knight", k=5)
recommend_collaborative(movie_id=1, user_id=1, k=5)