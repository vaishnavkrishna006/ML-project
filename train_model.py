import os
import pandas as pd
from src.models.recommender import MovieRecommender

def get_modern_collections():
    """
    Returns a dataframe of modern Marvel, DC, and WB movies.
    """
    modern_movies = [
        # Marvel (MCU)
        {"title": "Iron Man (2008)", "features": "Action Sci-Fi Adventure Marvel MCU"},
        {"title": "The Avengers (2012)", "features": "Action Sci-Fi Adventure Marvel MCU"},
        {"title": "Avengers: Endgame (2019)", "features": "Action Sci-Fi Adventure Marvel MCU"},
        {"title": "Guardians of the Galaxy (2014)", "features": "Action Sci-Fi Adventure Comedy Marvel MCU"},
        {"title": "Spider-Man: No Way Home (2021)", "features": "Action Adventure Sci-Fi Marvel MCU"},
        {"title": "Black Panther (2018)", "features": "Action Adventure Sci-Fi Marvel MCU"},
        
        # DC (DCEU & Nolan)
        {"title": "The Dark Knight (2008)", "features": "Action Crime Drama Thriller DC Batman"},
        {"title": "The Batman (2022)", "features": "Action Crime Drama Mystery DC"},
        {"title": "Man of Steel (2013)", "features": "Action Sci-Fi Adventure DC Superman"},
        {"title": "Wonder Woman (2017)", "features": "Action Adventure Fantasy DC"},
        {"title": "Justice League (2017)", "features": "Action Adventure Sci-Fi DC"},
        {"title": "Joker (2019)", "features": "Crime Drama Thriller DC"},
        
        # WB & Other Blockbusters
        {"title": "Inception (2010)", "features": "Action Sci-Fi Thriller WB Nolan"},
        {"title": "Interstellar (2014)", "features": "Sci-Fi Drama Adventure WB Nolan"},
        {"title": "Dune (2021)", "features": "Sci-Fi Adventure Drama WB"},
        {"title": "The Matrix (1999)", "features": "Action Sci-Fi WB"},
        {"title": "Mad Max: Fury Road (2015)", "features": "Action Adventure Sci-Fi WB"},
        {"title": "Harry Potter and the Sorcerer's Stone (2001)", "features": "Adventure Fantasy Family WB"},
    ]
    return pd.DataFrame(modern_movies)

def prepare_data():
    """
    Loads MovieLens 100k data and prepares features for content-based filtering.
    """
    item_path = 'data/ml-100k/u.item'
    if not os.path.exists(item_path):
        # If dataset is missing, we can still use our modern collections
        print(f"Warning: {item_path} not found. Using modern collections only.")
        return get_modern_collections()

    # MovieLens 100k genre list
    genres = [
        "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", 
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", 
        "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]
    
    # Load movies (u.item)
    cols = [1] + list(range(5, 24))
    names = ['title'] + genres
    
    movies = pd.read_csv(
        item_path, sep='|', encoding='latin-1', header=None, 
        usecols=cols, names=names
    )
    
    # Create 'features' string for each movie
    def get_features(row):
        movie_genres = [g for g in genres if row[g] == 1]
        return " ".join(movie_genres)

    print("Processing genres into features...")
    movies['features'] = movies.apply(get_features, axis=1)
    
    # Get standard dataset
    legacy_df = movies[['title', 'features']]
    
    # Get modern collections
    modern_df = get_modern_collections()
    
    # Merge them
    full_df = pd.concat([legacy_df, modern_df], ignore_index=True)
    
    return full_df

def main():
    print("Preparing dataset...")
    df = prepare_data()
    print(f"Dataset ready with {len(df)} movies.")
    
    recommender = MovieRecommender()
    recommender.fit(df)
    
    # Save the model
    artifact_path = "src/models/artifacts/recommender.pkl"
    print(f"Saving model to {artifact_path}...")
    recommender.save(artifact_path)
    print("Done!")

if __name__ == "__main__":
    main()
