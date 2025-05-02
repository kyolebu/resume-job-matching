import numpy as np
import json
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_similarity(resume_embeddings, job_embeddings):
    """
    Compute cosine similarity between resume and job embeddings
    
    Args:
        resume_embeddings: Array of resume embeddings
        job_embeddings: Array of job embeddings
        
    Returns:
        Similarity matrix
    """
    logger.info("Computing similarities...")
    
    # Normalize embeddings
    resume_embeddings = resume_embeddings / np.linalg.norm(resume_embeddings, axis=1, keepdims=True)
    job_embeddings = job_embeddings / np.linalg.norm(job_embeddings, axis=1, keepdims=True)
    
    # Compute cosine similarity
    similarity = np.dot(resume_embeddings, job_embeddings.T)
    
    return similarity

def find_top_matches(similarity, top_k=5):
    """
    Find top k matches for each resume
    
    Args:
        similarity: Similarity matrix
        top_k: Number of top matches to return
        
    Returns:
        Tuple of (indices of top matches, similarity scores)
    """
    logger.info(f"Finding top {top_k} matches...")
    
    # Get indices of top k matches for each resume
    top_indices = np.argsort(similarity, axis=1)[:, -top_k:]
    top_scores = np.take_along_axis(similarity, top_indices, axis=1)
    
    return top_indices, top_scores

def save_matches(top_indices, top_scores, output_file):
    """
    Save matches to a JSON file
    
    Args:
        top_indices: Array of top match indices
        top_scores: Array of similarity scores
        output_file: Path to save the matches
    """
    logger.info(f"Saving matches to {output_file}...")
    
    # Create output directory if it doesn't exist
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to list format for JSON serialization
    matches = []
    for resume_idx in range(len(top_indices)):
        resume_matches = []
        for match_idx, score in zip(top_indices[resume_idx], top_scores[resume_idx]):
            resume_matches.append({
                'job_index': int(match_idx),
                'similarity_score': float(score)
            })
        matches.append({
            'resume_index': resume_idx,
            'top_matches': resume_matches
        })
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(matches, f, indent=2)
    
    logger.info(f"Saved matches for {len(matches)} resumes") 