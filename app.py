import streamlit as st
import pandas as pd
from transformers import pipeline
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
GOOGLE_API_KEY = "AIzaSyDOedyZT5InL1M2CWdeJ3qFPQ65h05cK1o"
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the classifier
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def predict_genres(statement, genres):
    result = classifier(statement, genres)
    labels = result['labels']
    scores = result['scores']
    genre_scores = dict(zip(labels, scores))
    return genre_scores

def get_sorted_genres(statement):
    genres = [
        "Action", "Adventure", "Comedy", "Drama", "Thriller", "Horror",
        "Science Fiction (Sci-Fi)", "Fantasy", "Mystery", "Crime",
        "Romance", "Documentary", "Animation", "Superhero",
        "Psychological Thriller", "Historical", "War", "Western",
        "Musical", "Sports"
    ]
    
    genre_probs = predict_genres(statement, genres)
    sorted_genres = dict(sorted(genre_probs.items(), key=lambda item: item[1], reverse=True))
    return sorted_genres

def get_movie_recommendation(sorted_genres):
    try:
        # Get the top 3 genres
        top_genres = list(sorted_genres.keys())[:3]

        
        # Create the prompt
        prompt = f"""Based on the following genres: {", ".join(top_genres)}, 
        please recommend 3 movies that would be a good match. For each movie, include:
        1. Title
        2. Year
        3. Brief description
        4. Why it matches these genres
        
        Format the response in a clear, structured way."""
        
        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Generate response
        response = model.generate_content(prompt)
        
        return response.text
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        return None

# Streamlit app
st.set_page_config(
    page_title="Movie Recommendation App",
    page_icon="🎬",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .recommendation-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

st.title('🎬 Movie Recommendation App')
st.markdown("""
    Tell us what kind of movies you like, and we'll recommend some great films for you!
    For example: "I love exciting action movies with a good story"
""")

# User input
user_prompt = st.text_area('Enter your movie preferences:', height=100)

# Predict button
if st.button('Get Recommendations'):
    if user_prompt:
        with st.spinner('Analyzing your preferences and generating recommendations...'):
            sorted_genres = get_sorted_genres(user_prompt)
            #st.write(sorted_genres)
            recommendations = get_movie_recommendation(sorted_genres)
            print(recommendations)
            
            if recommendations:
                st.success('Here are your personalized movie recommendations:')
                st.write(recommendations)
            else:
                st.error('Failed to generate recommendations. Please try again.')
    else:
        st.warning('Please enter your movie preferences.')
