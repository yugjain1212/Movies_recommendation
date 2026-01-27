import pandas as pd
import pickle

def search_movies(movies_df, ratings_df, query, n_results=10):
    matches= movies_df[movies_df['title'].str.contains(query,case =False, na=False)]

    if len(matches)==0:
        print(f"No movies found matching '{query}'")
        return None

    movie_stats = ratings_df.groupby('movieId').agg({
        'rating': ['mean','count']
    }).reset_index()

    movie_stats.columns = ['movieId','avg_rating','num_ratings']

    results = matches.merge(movie_stats,on = 'movieId',how= 'left')
    results = results[['movieId','title','genres','avg_rating','num_ratings']]
    results['avg_rating'] = results['avg_rating'].fillna(0).round(2)
    results['num_ratings'] = results['num_ratings'].fillna(0).astype(int)
    results = results.sort_values('num_ratings',ascending= False).head(n_results)

    return results

def save_model(obj , filepath):
    with open(filepath,'wb') as f:
        pickle.dump(obj,f)
    print(f"saved to {filepath}")

def load_model(filepath):
    with open(filepath,'rb') as f:
        obj = pickle.load(f)
    print(f"loaded from {filepath}")
    return obj

def print_recommendations(recommendations_df, title = "Recommendations"):
    if recommendations_df is None or len(recommendations_df) ==0:
        print("None recommendations available")
        return
    print(f"\n{title}")
    print("*"*50)

    for idx, row in recommendations_df.iterrows():
        print(f"\n{idx+1}. {row['title']}")
        print(f" Genres : {row['genres']}")

        if 'similarity_score' in row:
            print(f"Similarity  : {row['similarity_score']:.3f}",end="")
        if 'predicted_rating' in row:
            print(f" Predicted Rating : {row['predicted_rating']:.2f}",end="")
        if 'avg_rating' in row :
            print(f" | Avg rating : {row['avg_rating']:.2f}",end = "")
        if 'num_ratings' in row:
            print(f" | Ratings {row['num_ratings']:,}",end = "")
        print()

def get_popular_movies(ratings_df,movies_df,n=20):
    popular = ratings_df.groupby('movieId').size().sort_values(ascending=False).head(n)

    popular_df = movies_df[movies_df['movieId'].isin(popular.index)].copy()
    popular_df['num_ratings'] = popular_df['movieId'].map(popular)
    popular_df = popular_df.sort_values('num_ratings',ascending=False)

    return popular_df[['movieId','title','genres','num_ratings']]

def get_top_rated_movies(ratings_df,movies_df,min_ratings =50,n=20):
    movie_stats = ratings_df.groupby('movieId').agg({
        'rating' : ['mean','count']
    }).reset_index()
    movie_stats.columns = ['movieId','avg_rating','num_ratings']

    top_rated = movie_stats[movie_stats['num_ratings']>=min_ratings]
    top_rated = top_rated.sort_values('avg_rated',ascending=False).head(n)

    # Merge with movie info
    result = top_rated.merge(movies_df[['movieId', 'title', 'genres']], on='movieId')
    result['avg_rating'] = result['avg_rating'].round(2)
    result['num_ratings'] = result['num_ratings'].astype(int)

    return result[['movieId', 'title', 'genres', 'avg_rating', 'num_ratings']]
