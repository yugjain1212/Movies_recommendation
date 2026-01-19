import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split


def load_raw_data (ratings_path,movies_path):
    ratings = pd.read_csv('/Users/yugjain/Documents/Machine_learning/Movies/datasets/raw/rating.csv')
    movies = pd.read_csv('/Users/yugjain/Documents/Machine_learning/Movies/datasets/raw/movie.csv')

    print(f"loaded  {len(ratings):,} ratings and {len(movies):,} movies")
    return ratings, movies

def sample_data(ratings, sample_fraction = 0.10,random_state=42):
    all_users = ratings['userId'].unique()
    sample_users= np.random.choice(all_users,size=int(len(all_users)*sample_fraction),replace = False)

    ratings_sample = ratings[ratings['userId'].isin(sample_users)].copy()
    print(f"sample {len(ratings_sample):,} ratings from {len(sample_users):,} users")
    return ratings_sample

def remove_missing_values(ratings,movies):

    ratings_clean = ratings.dropna(subset = ['userId','movieId','rating'])

    movies_clean = movies.dropna(subset= ['title'])
    if movies_clean['genres'].isnull().sum()>0:
        movies_clean['genres']=movies_clean['genres'].fillna('Unknown')

    print(f"Removed {len(ratings)-len(ratings_clean):,} rating with missing values")
    print(f"Removed {len(movies) - len(movies_clean):,} movies with missing values")

    return ratings_clean, movies_clean

def remove_duplicates(ratings,movies):
    ratings = ratings.sort_values('timestamp')
    ratings_clean = ratings.drop_duplicates(subset = ['userId','movieId'],keep ='last')

    movies_clean = movies.drop_duplicates(subset=['movieId'], keep='first')

    print(f"removed {len(ratings)-len(ratings_clean):,} duplicate ratings")
    print(f"removed {len(movies)- len(movies_clean):,} duplicate movies")

    return ratings_clean, movies_clean


def filter_low_activity(ratings, min_user_ratings =10,min_movie_ratings = 10):

    ratings_per_user = ratings.groupby('userId').size()
    active_users = ratings_per_user[ratings_per_user >= min_user_ratings].index

    ratings_per_movie = ratings.grouopby('movieId').size()
    popular_movies = ratings_per_movie[ratings_per_movie >=min_movie_ratings]

    ratings_filtered = ratings[
        ratings['userId'].isin(active_users) & ratings['movieId'].isin(popular_movies)

    ]
    print(f"filter to {len(active_users):,} active users(>= {min_user_ratings} ratings")
    print(f"Filtered to {len(popular_movies):,} popular movies (>={min_movie_ratings} ratings)")
    print(f"ratings after filtering : {len(ratings_filtered):.}")

    return ratings_filtered


def process_genres(movies):
    movies['genre_list']= movies['genres'].str.split('|')
    mlb = MultiLabelBinarizer()

    genre_matrix = mlb.fit_transform(movies['genre_list'])

    genre_df = pd.DataFrame(
        genre_matrix, columns= mlb.classes_, index= movies['movieId']
    )

    print(f"created genre matrixx : {genre_df.shape}")
    print(f"genres : {list(mlb.classes_)}")

    return genre_df

def normalize_ratings(ratings):
    user_mean_ratings = ratings.groupby('userId')['rating'].mean()

    ratings['user_mean']= ratings['userId'].map(user_mean_ratings)
    ratings['rating_normalized']= ratings['rating']-ratings['user_mean']

    print("Added normalized ratings")
    return ratings

def train_test_split_data(ratings,test_size = 0.2,random_state = 42):
    train, test = train_test_split(
        ratings,
        test_size=test_size,
        random_state=random_state
    )
    print(f"Train set: {len(train):,} ({len(train) / len(ratings) * 100:.2f}%)")
    print(f"Test set: {len(test):,} ({len(test) / len(ratings) * 100:.2f}%)")

    return train, test


def save_processed_data(ratings,movies,genre_matrix,train,test,output_dir='/Users/yugjain/Documents/Machine_learning/Movies/datasets'):


    os.makedirs(f'{output_dir}/processed', exist_ok=True)
    os.makedirs(f'{output_dir}/splits', exist_ok=True)

    ratings.to_csv(f'{output_dir}/processed/ratings_cleaned.csv', index=False)
    movies.to_csv(f'{output_dir}/processed/movies_cleaned.csv', index=False)
    genre_matrix.to_csv(f'{output_dir}/processed/genre_matrix.csv')

    train.to_csv(f'{output_dir}/splits/train_ratings.csv', index=False)
    test.to_csv(f'{output_dir}/splits/test_ratings.csv', index=False)

    print("\nSaved all processed data")


def preprocess_pipeline(ratings_path, movies_path,sample_fraction=None , min_user_ratings=10, min_movie_ratings = 10,test_size = 0.2,output_dir = '/Users/yugjain/Documents/Machine_learning/Movies/datasets',random_state = 42):
    print("*" *50)
    print("PREPROCESSING PIPELINE")
    print("*" *50)

    # Load data
    print("\n1. Loading data...")
    ratings, movies = load_raw_data(ratings_path, movies_path)
    # Sample if requested
    if sample_fraction is not None:
        print(f"\n2. Sampling {sample_fraction * 100:.0f}% of users...")
        ratings = sample_data(ratings, sample_fraction, random_state)
    # Remove missing values
    print("\n3. Removing missing values...")
    ratings, movies = remove_missing_values(ratings, movies)
    # Remove duplicates
    print("\n4. Removing duplicates...")
    ratings, movies = remove_duplicates(ratings, movies)

    # Filter low activity
    print("\n5. Filtering low-activity users and movies...")
    ratings = filter_low_activity(ratings, min_user_ratings, min_movie_ratings)

    # Filter movies to match ratings
    print("\n6. Filtering movies dataframe...")
    movies = movies[movies['movieId'].isin(ratings['movieId'].unique())].copy()
    movies = movies.reset_index(drop=True)
    print(f"Movies after filtering: {len(movies):,}")

    # Process genres
    print("\n7. Processing genres...")
    genre_matrix = process_genres(movies)

    # Normalize ratings
    print("\n8. Normalizing ratings...")
    ratings = normalize_ratings(ratings)

    # Train-test split
    print("\n9. Creating train-test split...")
    train, test = train_test_split_data(ratings, test_size, random_state)

    # Save everything
    print("\n10. Saving processed data...")
    save_processed_data(ratings, movies, genre_matrix, train, test, output_dir)

    print("\n" + "*" *50)
    print("PREPROCESSING COMPLETE!")
    print("=" * 70)
    print(f"\nFinal statistics:")
    print(f"  Users: {ratings['userId'].nunique():,}")
    print(f"  Movies: {len(movies):,}")
    print(f"  Ratings: {len(ratings):,}")
    print(f"  Genres: {genre_matrix.shape[1]}")
    print(f"  Train set: {len(train):,}")
    print(f"  Test set: {len(test):,}")
    return {
        'ratings': ratings,
        'movies': movies,
        'genre_matrix': genre_matrix,
        'train': train,
        'test': test
    }
