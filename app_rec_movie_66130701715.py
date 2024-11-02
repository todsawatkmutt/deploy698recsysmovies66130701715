import pickle
import streamlit as st
from surprise import SVD

# Load SVD model and data
with open('66130701715recommendation_movie_svd.pkl', 'rb') as file:
    svd_model, movie_ratings, movies = pickle.load(file)

# Streamlit interface
st.title("Movie Recommendation System")
st.write("Get top movie recommendations based on user preferences.")

# Input to select user ID
user_id = st.number_input("Enter User ID", min_value=1, step=1)

# Generate recommendations when button is clicked
if st.button("Show Recommendations"):
    # Get rated and unrated movies for the selected user
    rated_user_movies = movie_ratings[movie_ratings['userId'] == user_id]['movieId'].values
    unrated_movies = movies[~movies['movieId'].isin(rated_user_movies)]['movieId']

    # Predict ratings for unrated movies
    pred_rating = [svd_model.predict(user_id, movie_id) for movie_id in unrated_movies]
    sorted_predictions = sorted(pred_rating, key=lambda x: x.est, reverse=True)
    top_recommendations = sorted_predictions[:10]

    # Display recommendations
    st.subheader(f"Top 10 Movie Recommendations for User {user_id}")
    for recommendation in top_recommendations:
        movie_title = movies[movies['movieId'] == recommendation.iid]['title'].values[0]
        st.write(f"{movie_title} (Estimated Rating: {recommendation.est:.2f})")



