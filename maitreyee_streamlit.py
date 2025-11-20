import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os

# --- Configuration & Helper Functions ---

# Function to load and preprocess data (simulating your steps)
@st.cache_data
def load_and_preprocess_data():
    # Placeholder for actual data loading (you'd need to upload/access your CSVs)
    # Since I don't have the files, I'll simulate your final dataframes based on the operations.
    
    # In a real-world scenario, you would run all your previous notebook code here
    # to load, clean, and merge the dataframes (ratings, books, users).
    
    st.write("Simulating data loading and preprocessing... (Assumes CSVs are available)")
    
    # --- Data Loading and Cleaning (Simplified) ---
    # Replace with your actual file paths or uploaded files
    try:
        ratings = pd.read_csv("Ratings.csv")
        books = pd.read_csv("Books.csv")
        users = pd.read_csv("Users.csv")
    except:
        st.error("Error loading CSV files. Please ensure 'Ratings.csv', 'Books.csv', and 'Users.csv' are in the same directory as the app.py file.")
        return None, None, None, None, None

    # Cleaning Users (Age)
    users['Age'] = users['Age'].apply(lambda x: x if 5 <= x <= 100 else None)
    users['Age'].fillna(users['Age'].median(), inplace=True)
    
    # Cleaning Books (Year-Of-Publication, Author, Publisher)
    books['Year-Of-Publication'] = pd.to_numeric(books['Year-Of-Publication'], errors='coerce')
    books.loc[(books['Year-Of-Publication'] < 1800) | (books['Year-Of-Publication'] > 2022), 'Year-Of-Publication'] = np.nan
    books['Year-Of-Publication'].fillna(books['Year-Of-Publication'].median(), inplace=True)
    books['Book-Author'].fillna("Unknown", inplace=True)
    books['Publisher'].fillna("Unknown", inplace=True)
    
    # Merge for full data analysis
    ratings_books = pd.merge(ratings, books, on='ISBN', how='inner')
    full_data = pd.merge(ratings_books, users, on='User-ID', how='inner')

    # --- Popularity Based Prep ---
    book_rating_counts = full_data.groupby('Book-Title')['Book-Rating'].count().sort_values(ascending=False)
    avg_ratings = full_data.groupby('Book-Title')['Book-Rating'].mean()
    rating_summary = pd.DataFrame({
        'avg_rating': avg_ratings,
        'rating_count': book_rating_counts
    })
    
    # Filter for high-rated, popular books (>= 50 ratings)
    popular_high_rated = rating_summary[rating_summary['rating_count'] >= 50].sort_values(by='avg_rating', ascending=False)
    
    # --- Collaborative Filtering Prep ---
    # Filter for active users (>= 50 ratings) and popular books (>= 50 ratings)
    active_users = ratings['User-ID'].value_counts()
    active_users = active_users[active_users >= 50].index
    filtered_ratings = ratings[ratings['User-ID'].isin(active_users)]
    
    popular_books_cf = filtered_ratings['ISBN'].value_counts()
    popular_books_cf = popular_books_cf[popular_books_cf >= 50].index
    filtered_ratings = filtered_ratings[filtered_ratings['ISBN'].isin(popular_books_cf)]

    # Create user-book matrix
    user_book_matrix = filtered_ratings.pivot_table(
        index='User-ID', columns='ISBN', values='Book-Rating'
    ).fillna(0)
    
    # Compute cosine similarity
    book_similarity = cosine_similarity(user_book_matrix.T)
    book_similarity_df = pd.DataFrame(
        book_similarity, index=user_book_matrix.columns, columns=user_book_matrix.columns
    )
    
    # --- Hybrid/Content-Boosted Prep (for future model) ---
    # The final 'model_df' preparation (encoding and scaling) is typically done right before training.
    # We will return the necessary components to allow a user to potentially load a trained model later.

    return full_data, rating_summary, book_similarity_df, books, popular_high_rated


# Recommendation function for Item-Item CF
def recommend_books_cf(isbn, books_df, similarity_matrix, top_n=10):
    if isbn not in similarity_matrix.columns:
        return pd.DataFrame([{"Book-Title": "ISBN not found in the filtered matrix. Try a different book."}])
    
    # Get similarity scores for the given ISBN
    sim_scores = similarity_matrix[isbn]
    # Sort and remove the book itself
    similar_isbns = sim_scores.sort_values(ascending=False)[1:top_n+1]
    
    # Get book titles and details
    results = books_df[books_df["ISBN"].isin(similar_isbns.index)][["ISBN", "Book-Title", "Book-Author"]]
    
    # Create a Series for similarity scores to merge easily
    score_series = pd.Series(similar_isbns.values, index=similar_isbns.index)
    
    # Merge the scores back into the results DataFrame using index (which is ISBN)
    results = results.set_index("ISBN")
    results = results.join(score_series.rename("Similarity Score")).reset_index()
    
    return results.sort_values(by="Similarity Score", ascending=False)


# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="Book Recommender System")

st.title("📚 Advanced Book Recommendation System")
st.markdown("""
This application demonstrates three types of recommendation strategies based on the Book-Crossing dataset:
1. **Popularity-Based:** Recommends the most-rated or highest-rated books overall.
2. **Item-Item Collaborative Filtering:** Recommends books similar to a selected book based on user rating patterns.
3. **Hybrid/Content-Boosted:** A placeholder for a more advanced model (not fully implemented in the UI yet, but features were engineered).
""")

# Load data
full_data, rating_summary, book_similarity_df, books_df, popular_high_rated_df = load_and_preprocess_data()

if full_data is not None:

    # --- Tabbed Interface ---
    tab1, tab2 = st.tabs(["🔥 Popularity-Based Recommendations", "🤝 Collaborative Filtering"])

    with tab1:
        st.header("Overall Popularity Recommendations")
        st.subheader("Top 10 Most Rated Books")
        
        # Top 10 Most Rated
        most_rated_books = rating_summary.sort_values(by='rating_count', ascending=False).head(10)
        st.dataframe(most_rated_books.reset_index().rename(columns={'Book-Title': 'Title', 'rating_count': 'Total Ratings', 'avg_rating': 'Average Rating'})[['Title', 'Total Ratings', 'Average Rating']], use_container_width=True)

        st.subheader("Top 10 Highest Rated Popular Books (Min 50 Ratings)")
        # Top 10 Highest Rated Popular Books
        st.dataframe(popular_high_rated_df.head(10).reset_index().rename(columns={'Book-Title': 'Title', 'rating_count': 'Total Ratings', 'avg_rating': 'Average Rating'})[['Title', 'Total Ratings', 'Average Rating']], use_container_width=True)


    with tab2:
        st.header("Item-Item Collaborative Filtering")
        st.markdown("Enter the ISBN of a popular book to get similar book recommendations based on shared user ratings.")
        
        # Get list of popular ISBNs (for selection/testing)
        popular_isbns = book_similarity_df.columns.tolist()
        
        # Create a mapping from ISBN to Title for user-friendly selection
        isbn_to_title = books_df[books_df['ISBN'].isin(popular_isbns)].set_index('ISBN')['Book-Title'].to_dict()
        title_to_isbn = {v: k for k, v in isbn_to_title.items()}
        
        # Pre-select ISBN for the example book "Harry Potter and the Order of the Phoenix" (043935806X)
        default_isbn = "043935806X"
        default_title = isbn_to_title.get(default_isbn, "Select a Book")
        
        # User selection for the book
        selected_title = st.selectbox(
            "Select a Book Title to Find Similar Books",
            options=[default_title] + sorted(title_to_isbn.keys()),
            index=0 # Default to the example book
        )
        
        # Get the actual ISBN for the selected title
        selected_isbn = title_to_isbn.get(selected_title, default_isbn)
        
        if st.button("Get Recommendations"):
            if selected_isbn:
                with st.spinner(f"Finding similar books for **{selected_title}**..."):
                    recommendations = recommend_books_cf(
                        selected_isbn, books_df, book_similarity_df, top_n=10
                    )
                    
                    st.subheader(f"Top 10 Books Similar to: **{selected_title}**")
                    
                    # Merge book details for display
                    display_df = recommendations.merge(books_df[['ISBN', 'Year-Of-Publication', 'Publisher']], on='ISBN', how='left')
                    
                    st.dataframe(
                        display_df[['Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Similarity Score']].rename(
                            columns={'Book-Title': 'Title', 'Book-Author': 'Author', 'Year-Of-Publication': 'Year', 'Publisher': 'Publisher'}
                        ),
                        use_container_width=True
                    )
            else:
                st.warning("Please select a book to get recommendations.")

    # --- Footer ---
    st.markdown("---")
    st.caption(f"Total Records Analyzed: {len(full_data):,} | Total Unique Users: {full_data['User-ID'].nunique():,} | Total Unique Books: {full_data['ISBN'].nunique():,}")

