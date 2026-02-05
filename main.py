#MOVIE RECOMMENDATION SYSTEM - MAIN INTERFACE
""" A HYBRID RECOMMENDATION SYSTEM COMBINING CONTENT BASED AND COLLABORATIVE FILTERING """

import sys
import os
from operator import truediv

import pandas as pd
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__),'src'))

from src.data_preprocessing import preprocess_pipeline
from src.content_based import  ContentBasedRecommender
from src.collaborative import CollaborativeRecommender
from src.utils import search_movies,print_recommendations,get_popular_movies,get_top_rated_movies
from src.evaluation import  evaluate_collaborative_filtering,print_evaluation

class MovieRecommendationSystem:
    def __init__(self,data_dir='datasets',models_dir ='models'):

        self.data_dir = data_dir
        self.models_dir = models_dir

        self.movies = None
        self.ratings = None
        self.train = None
        self.test = None
        self.genre_matrix = None

        self.content_recommender = None
        self.collaborative_recommender =None

        print("*"*50)
        print("Movie recommendation System")
        print("*"*50)

    def load_data(self):
        print("\nLoading preprocessed data..")

        try:
            self.movies = pd.read_csv(f'{self.data_dir}/processed/movies_cleaned.csv')
            self.ratings = pd.read_csv(f'{self.data_dir}/processed/ratings_cleaned.csv')
            self.train = pd.read_csv(f'{self.data_dir}/splits/train_ratings.csv')
            self.test = pd.read_csv(f'{self.data_dir}/splits/test_ratings.csv')
            self.genre_matrix = pd.read_csv(f'{self.data_dir}/processed/genre_matrix.csv', index_col=0)

            print(f"Loaded {len(self.movies):,} movies")
            print(f"loaded {len(self.ratings):,} ratings")
            print(f"Train set {len(self.train):,}")
            print(f"test set :{len(self.test):,}")
            return True

        except FileNotFoundError as e:
            print(f"\n Error preprocessed data not found")
            print(f"{e}")
            print(f"\nplease run preprocessing first :")
            print(" python main.py  --preprocess")
            return False

    def preprocess_data(self,sample_fraction =None):
        print("\nStarting preprocessing pipeline...")

        ratings_path = f'{self.data_dir}/raw/ratings.csv'
        movies_path = f'{self.data_dir}/raw/movies.csv'

        if not os.path.exists(ratings_path) or not os.path.exists(movies_path):
            print("\n error raw data files not found")
            print(f"expected : {ratings_path}")
            print(f"expected {ratings_path}")
            print("\nPlease download the MovieLens dataset and place it in data/raw/")
            return False

        result = preprocess_pipeline(
            ratings_path = ratings_path,
            movies_path = movies_path,
            sample_fraction=sample_fraction,
            min_user_ratings=10,
            min_movie_ratings=10,
            test_size=0.2,
            output_dir =self.data_dir,
            random_state = 42
        )
        self.movies = result['movies']
        self.ratings  =result['ratings']
        self.train = result['train']
        self.test = result['test']
        self.genre_matrix = result['genre_matrix']

        return True

    def build_content_based(self):
        print("Building content-based recommender...")

        # Initialize content-based recommender
        self.content_recommender = ContentBasedRecommender(
            self.movies, self.ratings, self.genre_matrix
        )
        # Calculate and save similarity matrix
        self.content_recommender.calculate_similarity()
        os.makedirs(self.models_dir, exist_ok=True)
        self.content_recommender.save_model(os.path.join(self.models_dir, 'content_similarity_matrix.pkl'))

        print("Content-based recommender ready")

    def build_collaborative(self):
        print("Building collaborative filtering recommender...")

        self.collaborative_recommender = CollaborativeRecommender(
            self.movies,
            self.ratings,
            self.train
        )
        self.collaborative_recommender.create_user_item_matrix()
        self.collaborative_recommender.calculate_item_similarity()
        self.collaborative_recommender.calculate_user_similarity()

        os.makedirs(self.models_dir, exist_ok=True)
        self.collaborative_recommender.save_models(self.models_dir)

        print("Collaborative filtering recommender ready")

    def load_models(self):
        print("\nLoading pre-trained Models...")

        try:
            # Initialize recommenders
            self.content_recommender = ContentBasedRecommender(
                self.movies,
                self.ratings,
                self.genre_matrix
            )
            self.collaborative_recommender = CollaborativeRecommender(
                self.movies,
                self.ratings,
                self.train
            )

            # Load models from disk
            content_model_path = os.path.join(self.models_dir, 'content_similarity_matrix.pkl')
            self.content_recommender.load_model(content_model_path)
            self.collaborative_recommender.load_models(self.models_dir)

            print("Models loaded successfully")
            return True

        except FileNotFoundError:
            print("\nPre-trained models not found")
            print("Building models from scratch...")
            self.build_content_based()
            self.build_collaborative()
            return True


    def search(self,query,n=10):
        results = search_movies(self.movies,self.ratings,query,n)

        if results is not None :
            print(f"\n Search results for '{query}' :")
            print(results.to_string(index = False))

        return results

    def recommend_content_based(self,movie_title,n=10,min_ratings =20):
        print("*"*50)
        print("CONTENT BASED RECOMMENDATIONS")
        print("Based on genres similarity")
        print("*"*50)

        recs = self.content_recommender.get_recommendations(
            movie_title,
            n_recommendations=n,
            min_ratings = min_ratings
        )
        if recs is not None:
            print_recommendations(recs,"Recommendations")

        return recs


    def recommend_collaborative(self,movie_title,n=10,min_ratings =20):
        print("*"*50)
        print("COLLABORATIVE FILTERING RECOMMENDATIONS")
        print("Based on user rating patterns")
        print("*"*50)

        recs = self.collaborative_recommender.get_item_based_recommendations(
            movie_title,
            n_recommendations=n,
            min_ratings = min_ratings
        )
        if recs is not None:
            print_recommendations(recs,"Recommendations")

        return recs

    def recommend_for_user(self,user_id,n=10):
        print("*"*50)
        print(f"Personalized recommendations for User {user_id}")
        print("*"*50)

        recs = self.collaborative_recommender.get_user_based_recommendations(
            user_id, n_recommendations=n
        )

        if recs is not None:
            print_recommendations(recs,"Recommendations")

            print("*"*50)
            print(f"Movie user {user_id} has already rated(top 10):")
            print("*"*50)

            user_ratings = self.ratings[self.ratings['userId']==user_id].merge(
                self.movies[['movieId','title','genres']], on= 'movieId'
            ).sort_values('rating',ascending=False).head(10)

            for i,row in user_ratings.iterrows():
                print(f"{row['title']}")
                print(f" Rating: {row['rating']:.2f} | Genres :{row['genres']}\n")
            return recs

        def compare_methods(self,movie_title,n=5):
            print("*"*50)
            print(f"COMPARING THE METHODS FOR {movie_title}")
            print("*"*50)

            print("Content based {genre similarity}")
            content_recs = self.content_recommender.get_recommendations(movie_title,n_recommendations=n)
            if content_recs is not None:
                for idx, row in content_recs.iterrows():
                    print(f"{idx + 1}. {row['title']}")
                    print(f"   Genres: {row['genres']}")
                    print(f"   Similarity: {row['similarity_score']:.3f}\n")

            print("\n Collaborative Filtering (User behaviour)")
            collab_recs = self.collaborative.get_item_based_recommendations(movie_title,n_recommendations=n)
            if collab_recs is not None:
                for idx, row in collab_recs.iterrows():
                    print(f"{idx + 1}. {row['title']}")
                    print(f"   Genres: {row['genres']}")
                    print(f"   Similarity: {row['similarity_score']:.3f}\n")

            print("\nKEY DIFFERENCES:")
            print("   Content-based: Recommends similar genres")
            print("   Collaborative: May cross genres based on user patterns")

        def evaluate(self,sample_size=1000):
            print("*"*50)
            print("Evaluating recommendations system")
            print("*"*50)

            metrics = evaluate_collaborative_filtering(
                self.test,
                self.collaborative_recommender.create_user_item_matrix,
                self.collaborative_recommender.item_similarity,
                sample_size = sample_size,
                k=10
            )
            if metrics:
                print_evaluation(metrics)
                print(f"Coverage : {metrics['coverage']:.2f}%")

        def show_popular_movies(self,n=20):
            print(f"Top {n} most popular Movies")

            popular = get_popular_movies(self.ratings,self.movies,n)
            print(popular.to_string(index=False))


        def interactive_mode(self):
            print("\n" + "*"*70)
            print("INTERACTIVE MODE")
            print("="*70)
            print("\nCommands:")
            print("  search <query>           - Search for movies")
            print("  content <movie title>    - Content-based recommendations")
            print("  collab <movie title>     - Collaborative recommendations")
            print("  user <user_id>           - Personalized recommendations")
            print("  compare <movie title>    - Compare both methods")
            print("  popular                  - Show popular movies")
            print("  top                      - Show top rated movies")
            print("  evaluate                 - Evaluate system performance")
            print("  help                     - Show this help")
            print("  quit                     - Exit")
            while True:
                try:
                    command = input("\n>>> ").strip()
                    if not command:
                        continue
                    parts = command.split(maxsplit=1)
                    cmd = parts[0].lower()
                    arg = parts[1] if len(parts) > 1 else None
                    if cmd == 'quit' or cmd == 'exit':
                        print("\nGoodbye!")
                        break
                    elif cmd == 'help':
                         print("\nCommands:")
                         print("  search <query>")
                         print("  content <movie title>")
                         print("  collab <movie title>")
                         print("  user <user_id>")
                         print("  compare <movie title>")
                         print("  popular")
                         print("  top")
                         print("  evaluate")
                         print("  quit")

                    elif cmd == 'search':
                        if arg:
                         self.search(arg)
                        else:
                           print("Usage: search <query>")
                    elif cmd == 'content':
                               if arg:
                                   self.recommend_content_based(arg)
                               else:
                                  print("Usage: content <movie title>")
                    elif cmd == 'collab':
                            if arg:
                                self.recommend_collaborative(arg)
                            else:
                               print("Usage: collab <movie title>")
                    elif cmd == 'user':
                           if arg:
                                 try:
                                       user_id = int(arg)
                                       self.recommend_for_user(user_id)
                                 except ValueError:
                                       print("User ID must be a number")
                           else:
                                print("Usage: user <user_id>")
                    elif cmd == 'compare':
                           if arg:
                              self.compare_methods(arg)
                           else:
                               print("Usage: compare <movie title>")
                    elif cmd == 'popular':
                                self.show_popular_movies()

                    elif cmd == 'top':
                              self.show_top_rated_movies()
                    elif cmd == 'evaluate':
                             self.evaluate()

                    else:
                       print(f"Unknown command: {cmd}")
                       print("Type 'help' for available commands")

                except KeyboardInterrupt:
                    print("\n\nGoodbye!")
                    break
                except Exception as e:
                    print(f"\n❌ Error: {e}")

def main():
    """
    Main function with command-line arguments
    """
    parser = argparse.ArgumentParser(
        description='Movie Recommendation System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
  # Preprocess data (first time setup)
  python main.py --preprocess
  
  # Preprocess with sampling (for testing)
  python main.py --preprocess --sample 0.1
  
  # Build models
  python main.py --build
  
  # Search for a movie
  python main.py --search "batman"
  
  # Get content-based recommendations
  python main.py --content "Toy Story (1995)"
  
  # Get collaborative recommendations
  python main.py --collab "Inception (2010)"
  
  # Get personalized recommendations for a user
  python main.py --user 123
  
  # Compare methods
  python main.py --compare "The Matrix (1999)"
  
  # Interactive mode
  python main.py --interactive
  
  # Evaluate system
  python main.py --evaluate
        """
    )

    # Arguments
    parser.add_argument('--preprocess', action='store_true',
                        help='Run data preprocessing')
    parser.add_argument('--sample', type=float,
                        help='Sample fraction of users (e.g., 0.1 for 10%%)')
    parser.add_argument('--build', action='store_true',
                        help='Build recommendation models')
    parser.add_argument('--search', type=str,
                        help='Search for movies')
    parser.add_argument('--content', type=str,
                        help='Get content-based recommendations')
    parser.add_argument('--collab', type=str,
                        help='Get collaborative filtering recommendations')
    parser.add_argument('--user', type=int,
                        help='Get personalized recommendations for user ID')
    parser.add_argument('--compare', type=str,
                        help='Compare recommendation methods for a movie')
    parser.add_argument('--interactive', action='store_true',
                        help='Start interactive mode')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate recommendation system')
    parser.add_argument('--popular', action='store_true',
                        help='Show popular movies')
    parser.add_argument('--top', action='store_true',
                        help='Show top rated movies')

    args = parser.parse_args()

    # Initialize system
    system = MovieRecommendationSystem()

    # Preprocess
    if args.preprocess:
        success = system.preprocess_data(sample_fraction=args.sample)
        if not success:
            return
        if args.build:
            system.build_content_based()
            system.build_collaborative()
        return

    # Load data
    if not system.load_data():
        return

    # Build models
    if args.build:
        system.build_content_based()
        system.build_collaborative()
        return

    # Load models
    system.load_models()

    # Execute commands
    if args.search:
        system.search(args.search)

    elif args.content:
        system.recommend_content_based(args.content)

    elif args.collab:
        system.recommend_collaborative(args.collab)

    elif args.user:
        system.recommend_for_user(args.user)

    elif args.compare:
        system.compare_methods(args.compare)

    elif args.evaluate:
        system.evaluate()

    elif args.popular:
        system.show_popular_movies()

    elif args.top:
        system.show_top_rated_movies()

    elif args.interactive:
        system.interactive_mode()

    else:
        # No arguments - show help
        parser.print_help()
        print("\n💡 Tip: Try running:")
        print("   python main.py --interactive")


if __name__ == "__main__":
    main()






















