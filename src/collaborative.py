import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle

class CollaborativeRecommender:

    def __init__(self,movies_df,ratings_df,train_df):
        self.movies = movies_df
        self.ratings = ratings_df
        self.train = train_df

        self.user_item_matrix = None
        self.item_similarity = None
        self.user_similarity = None


        self.movie_stats = self.ratings.groupby('movieId').agg({
            'rating' : ['mean','count']
        }).reset_index()
        self.movie_stats.columns = ['movieId','avg_rating','num_ratings']


    def create_user_item_matrix(self):
        self.user_item_matrix = self.train.pivot_table(
            index='userId',
            columns='movieId',
            values='rating'
        )
        total_cells = self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1]
        filled_cells = self.user_item_matrix.count().sum()
        sparsity = (1-filled_cells/total_cells)*100

        print(f"user_item matrix {self.user_item_matrix.shape}")
        print(f"sparsity : {sparsity:.2f}%")

        return self.user_item_matrix

    def calculate_item_similarity(self):
        if self.user_item_matrix is None:
            raise ValueError("User-item matrix not created. Run create_user_item_matrix() first.")
        print("Calculating item-based similarity...")
        # Fill NaN with 0
        filled = self.user_item_matrix.fillna(0)
        # Calculate similarity (transpose so movies are rows)
        similarity = cosine_similarity(filled.T)
        self.item_similarity = pd.DataFrame(
            similarity,
            index=self.user_item_matrix.columns,
            columns=self.user_item_matrix.columns
        )
        print(f"Item similarity matrix: {self.item_similarity.shape}")
        return self.item_similarity

    def calculate_user_similarity(self):
        if self.user_item_matrix is None:
            raise ValueError("User-item matrix not created. Run create_user_item_matrix() first.")
        print("Calculating user-based similarity...")
        # Fill NaN with 0
        filled = self.user_item_matrix.fillna(0)
        similarity = cosine_similarity(filled)
        self.user_similarity = pd.DataFrame(
            similarity,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )
        print(f"User similarity matrix: {self.user_similarity.shape}")
        return self.user_similarity
    def get_item_based_recommendations(self,movie_title,n_recommendations =10,min_ratings =20):
        if self.item_similarity is None:
            raise ValueError("Item similarity not calculated. Run calculate_item_similarity() first.")
        movie_match = self.movies[self.movies['title'] == movie_title]
        if len(movie_match) == 0:
            print(f"Movie '{movie_title}' not found")
            return None
        movie_id = movie_match['movieId'].values[0]
        if movie_id not in self.item_similarity.columns:
            print(f"Movie ID {movie_id} not in similarity matrix")
            return None
        movie_genres = movie_match['genres'].values[0]
        print(f"Finding recommendations for: {movie_title}")
        print(f"Genres: {movie_genres}")
        print(f"Method: Item-based collaborative filtering\n")

        similar_scores = self.item_similarity[movie_id].sort_values(ascending=False)
        similar_scores = similar_scores[similar_scores.index != movie_id]

        # Get candidates
        top_candidates = similar_scores.head(n_recommendations * 3)

        # Build recommendations
        recommendations = []
        for candidate_id, similarity_score in top_candidates.items():
            movie_info_df = self.movies[self.movies['movieId'] == candidate_id]
            if len(movie_info_df) == 0:
                continue

            movie_info = movie_info_df.iloc[0]
            stats = self.movie_stats[self.movie_stats['movieId'] == candidate_id]

            if len(stats) > 0:
                avg_rating = stats['avg_rating'].values[0]
                num_ratings = stats['num_ratings'].values[0]

                if num_ratings >= min_ratings:
                    recommendations.append({
                        'movieId': candidate_id,
                        'title': movie_info['title'],
                        'genres': movie_info['genres'],
                        'similarity_score': round(similarity_score, 3),
                        'avg_rating': round(avg_rating, 2),
                        'num_ratings': int(num_ratings)
                    })
        recommendations_df = pd.DataFrame(recommendations)
        recommendations_df = recommendations_df.sort_values(
            ['similarity_score', 'avg_rating'],
            ascending=[False, False]
        ).head(n_recommendations)

        return recommendations_df

    def get_user_based_recommendations(self, user_id, n_recommendations=10, n_similar_users=10):
        if self.user_similarity is None:
            raise ValueError("User similarity not calculated. Run calculate_user_similarity() first.")

        if user_id not in self.user_similarity.index:
            print(f"User {user_id} not found")
            return None

        print(f"Finding Recommendations for user {user_id}")
        print(f"method user-base collabrative filtering\n")

        similar_users = self.user_similarity[user_id].sort_values(ascending=False)
        similar_users = similar_users[similar_users.index != user_id]
        top_similar_users = similar_users.head(n_similar_users)

        user_rated_movies = set(self.user_item_matrix.loc[user_id].dropna().index)

        recommendations_dict = {}

        for similar_user_id, similarity_score in top_similar_users.items():
            similar_user_ratings = self.user_item_matrix.loc[similar_user_id].dropna()

            for movie_id, rating in similar_user_ratings.items():
                if movie_id in user_rated_movies:
                    continue

                if movie_id not in recommendations_dict:
                    recommendations_dict[movie_id] = {
                        'weighted_sum': 0,
                        'similarity_sum': 0
                    }

                recommendations_dict[movie_id]['weighted_sum'] += rating * similarity_score
                recommendations_dict[movie_id]['similarity_sum'] += similarity_score

        predicted_ratings = []
        for movie_id, data in recommendations_dict.items():
            if data['similarity_sum'] > 0:
                predicted_rating = data['weighted_sum'] / data['similarity_sum']
                predicted_ratings.append({
                    'movieId': movie_id,
                    'predicted_rating': predicted_rating
                })

        predictions_df = pd.DataFrame(predicted_ratings)
        predictions_df = predictions_df.sort_values('predicted_rating', ascending=False)
        top_predictions = predictions_df.head(n_recommendations)

        recommendations = []
        for _, row in top_predictions.iterrows():
            movie_id = row['movieId']
            movie_info_df = self.movies[self.movies['movieId'] == movie_id]

            if len(movie_info_df) == 0:
                continue

            movie_info = movie_info_df.iloc[0]
            stats = self.movie_stats[self.movie_stats['movieId'] == movie_id]

            if len(stats) > 0:
                recommendations.append({
                    'movieId': movie_id,
                    'title': movie_info['title'],
                    'genres': movie_info['genres'],
                    'predicted_rating': round(row['predicted_rating'], 2),
                    'avg_rating': round(stats['avg_rating'].values[0], 2),
                    'num_ratings': int(stats['num_ratings'].values[0])
                })

        return pd.DataFrame(recommendations)

    def save_models(self, output_dir='/Users/yugjain/Documents/Machine_learning/Movies/models'):
        import os
        os.makedirs(output_dir, exist_ok=True)

        if self.item_similarity is not None:
            with open(f'{output_dir}/item_similarity_matrix.pkl', 'wb') as f:
                pickle.dump(self.item_similarity, f)
            print(f"Saved item similarity matrix")

        if self.user_similarity is not None:
            with open(f'{output_dir}/user_similarity_matrix.pkl', 'wb') as f:
                pickle.dump(self.user_similarity, f)
            print(f"Saved user similarity matrix")

        if self.user_item_matrix is not None:
            with open(f'{output_dir}/user_item_matrix.pkl', 'wb') as f:
                pickle.dump(self.user_item_matrix, f)
            print(f"Saved user-item matrix")

    def load_models(self, output_dir='/Users/yugjain/Documents/Machine_learning/Movies/models'):

        try:
            with open(f'{output_dir}/item_similarity_matrix.pkl', 'rb') as f:
                self.item_similarity = pickle.load(f)
            print(f"✓ Loaded item similarity matrix")
        except:
            print("Item similarity matrix not found")

        try:
            with open(f'{output_dir}/user_similarity_matrix.pkl', 'rb') as f:
                self.user_similarity = pickle.load(f)
            print(f"✓ Loaded user similarity matrix")
        except:
            print("User similarity matrix not found")

        try:
            with open(f'{output_dir}/user_item_matrix.pkl', 'rb') as f:
                self.user_item_matrix = pickle.load(f)
            print(f"✓ Loaded user-item matrix")
        except:
            print("User-item matrix not found")

