import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle


class ContentBasedRecommender :
    def __init__(self,movies_df,ratings_df,genre_matrix):

        self.movies=movies_df
        self.ratings = ratings_df
        self.genre_matrix = genre_matrix
        self.similarity_matrix = None

        self.movie_stats = self.ratings.groupby('movieId').agg({'rating' : ['mean','count']}).reset_index()
        self.movie_stats.columns = ['movieId','avg_rating','num_ratings']

    def calculate_similarity(self):
        print("Calculating the movie similarity matrix..")

        similarity = cosine_similarity(self.genre_matrix)

        self.similarity_matrix = pd.DataFrame(
            similarity,index = self.genre_matrix.index,columns= self.genre_matrix.index

        )
        print(f"Similarity matrix calculated : {self.similarity_matrix.shape}")
        return self.similarity_matrix

    def get_recommendations(self,movie_title,n_recommendations =10,min_ratings = 20):
        if self.similarity_matrix is None:
            raise ValueError("similarity matrix not calculated. run calculate_similarity() first.")

        movie_match = self.movies[self.movies['title']==movie_title]

        if len(movie_match)==0:
            print(f"movie ' {movie_title}' not found")
            return self._search_movies(movie_title)

        movie_id=movie_match['movieId'].values[0]

        if movie_id not in self.similarity_matrix.index:
            print(f"Movie ID {movie_id} not in similarity matrix")
            return None

        movie_genres = movie_match['genres'].values[0]

        print(f"Finding recommendations for: {movie_title}")
        print(f"Genres: {movie_genres}\n")

        # Get similar movies
        similar_scores = self.similarity_matrix.loc[movie_id].sort_values(asecending = False)
        similar_scores = similar_scores[similar_scores.index != movie_id]

        top_candidate = similar_scores.head(n_recommendations*3)

        recommendations = []
        for candidate_id ,similarity_score in top_candidate.items():
            movie_info_df = self.movies[self.movies['movieId']==candidate_id]

            if len(movie_info_df) ==0:
                continue

            movie_info = movie_info_df.iloc[0]
            stats = self.movie_stats[self.movie_stats['movieId']==candidate_id]

            if len(stats)>0:
                avg_rating = stats['avg_rating'].values[0]
                num_ratings = stats['num_ratings'].values[0]

                if num_ratings >= min_ratings :
                    recommendations.append({
                        'movieId' :candidate_id,
                        'title' : movie_info['title'],
                        'genres' : movie_info['genres'],
                        'similarity_score' : round(similarity_score,3),
                        'avg_rating' : round(avg_rating,2),
                        'num_ratings' : int(num_ratings)
                    })
        recommendations_df = pd.DataFrame(recommendations)
        recommendations_df = recommendations_df.sort_values(
            ['similarity_score','avg_rating'], ascending=[False,False]
        ).head(n_recommendations)

        return recommendations_df

    def _search_movies(self,query,n_results = 10):
        matches = self.movies[self.movies['title'].str.contains(query,case =False,na=False)]

        if len(matches) ==0:
            print(f"No movies found matching '{query}'")
            return None

        results = matches.merge(self.movie_stats,on = 'movieId',how='left')
        results = results[['movieId','title','genres','avg_rating','num_ratings']]
        results = results.sort_values('num_ratings',ascending =False).head(n_results)

        print(f"\nDid you mean one of these ?")
        return results

    def save_model(self,file_path = '/Users/yugjain/Documents/Machine_learning/Movies/models/content_similarity_matrix.pkl'):
        if self.similarity_matrix is None:
            raise ValueError("No similarity matrix to save")

        with open(file_path,'wb')as f:
            pickle.dump(self.similarity_matrix,f)

        print(f"Saved similarity matrix to {file_path}")


    def load_model(self, file_path ='/Users/yugjain/Documents/Machine_learning/Movies/models/content_similarity_matrix.pkl'):
        with open(file_path,'rb') as f:
            self.similarity_matrix = pickle.load(f)

        print(f"Loaded similarity matrix from {file_path}")



