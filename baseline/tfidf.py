import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import save_npz

from utils.data_loader import load_processed_data
from utils.matching import compute_similarity, find_top_matches, save_matches

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_tfidf_matches(resumes_df, jobs_df, top_k=5):
    """
    Compute TF-IDF embeddings and find top matches
    
    Args:
        resumes_df: DataFrame containing resumes
        jobs_df: DataFrame containing jobs
        top_k: Number of top matches to return per resume
        
    Returns:
        List of dictionaries containing matches and scores
    """
    logger.info("Computing TF-IDF similarities...")
    
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=10000)
    
    # Combine all text for fitting
    all_text = pd.concat([
        resumes_df['processed_text'],
        jobs_df['combined_text']
    ])
    
    # Fit and transform
    logger.info("Fitting TF-IDF vectorizer...")
    vectorizer.fit(all_text)
    
    # Transform resumes and jobs separately
    logger.info("Transforming resumes and jobs...")
    resume_vectors = vectorizer.transform(resumes_df['processed_text'])
    job_vectors = vectorizer.transform(jobs_df['combined_text'])
    
    # Save embeddings
    logger.info("Saving embeddings...")
    Path('./data/embeddings').mkdir(parents=True, exist_ok=True)
    save_npz('./data/embeddings/tfidf_resume_embeddings.npz', resume_vectors)
    save_npz('./data/embeddings/tfidf_job_embeddings.npz', job_vectors)
    
    # Compute similarities in batches to save memory
    logger.info("Computing similarities and finding top matches...")
    batch_size = 100
    matches = []
    
    for i in range(0, len(resumes_df), batch_size):
        # Get batch of resume vectors
        batch_vectors = resume_vectors[i:i+batch_size]
        
        # Compute similarities for this batch
        batch_similarities = cosine_similarity(batch_vectors, job_vectors)
        
        # Find top matches for each resume in batch
        for j, similarities in enumerate(batch_similarities):
            resume_idx = i + j
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            top_scores = similarities[top_indices]
            
            # Store matches for this resume
            resume_matches = []
            for job_idx, score in zip(top_indices, top_scores):
                resume_matches.append({
                    'job_index': int(job_idx),
                    'similarity_score': float(score)
                })
            
            matches.append({
                'resume_index': resume_idx,
                'top_matches': resume_matches
            })
            
        logger.info(f"Processed {len(matches)} resumes...")
    
    return matches

def main():
    # Load pre-processed data
    resumes_df, jobs_df = load_processed_data(return_dataframe=True)
    
    # Compute TF-IDF matches
    matches = compute_tfidf_matches(resumes_df, jobs_df)
    
    # Save results
    save_matches(matches, './data/matches/tfidf_matches.json')
    
    # Print example matches
    logger.info("\nExample matches for first resume:")
    for i, match in enumerate(matches[0]['top_matches']):
        logger.info(f"Match {i+1}: Job {match['job_index']} (similarity: {match['similarity_score']:.4f})")

if __name__ == "__main__":
    main() 