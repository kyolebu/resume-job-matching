import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from utils.data_loader import load_processed_data
from utils.matching import compute_similarity, find_top_matches, save_matches

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_embeddings(texts, model_name='all-MiniLM-L6-v2', batch_size=32):
    """
    Generate Sentence BERT embeddings for texts using Sentence Transformers
    
    Args:
        texts: List of texts to embed
        model_name: Name of the Sentence Transformer model to use
        batch_size: Batch size for processing
        
    Returns:
        Numpy array of embeddings
    """
    logger.info(f"Loading Sentence Transformer model {model_name}...")
    model = SentenceTransformer(model_name)
    
    logger.info("Generating embeddings...")
    embeddings = []
    
    # Process in batches to save memory
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        embeddings.append(batch_embeddings)
    
    return np.vstack(embeddings)

def main():
    # Load pre-processed data
    resume_texts, job_texts = load_processed_data(return_dataframe=False)
    
    # Generate embeddings
    logger.info("Generating resume embeddings...")
    resume_embeddings = generate_embeddings(resume_texts)
    
    logger.info("Generating job embeddings...")
    job_embeddings = generate_embeddings(job_texts)
    
    # Save embeddings
    logger.info("Saving embeddings...")
    np.savez('./data/embeddings/sentence_bert_resume_embeddings.npz', embeddings=resume_embeddings)
    np.savez('./data/embeddings/sentence_bert_job_embeddings.npz', embeddings=job_embeddings)
    
    # Compute similarities
    similarity = compute_similarity(resume_embeddings, job_embeddings)
    
    # Find top matches
    top_indices, top_scores = find_top_matches(similarity)
    
    # Save matches
    save_matches(top_indices, top_scores, './data/matches/sentence_bert_matches.json')
    
    # Print example matches
    logger.info("\nExample matches for first resume:")
    for i, (job_idx, score) in enumerate(zip(top_indices[0], top_scores[0])):
        logger.info(f"Match {i+1}: Job {job_idx} (similarity: {score:.4f})")
    
    logger.info("Processing complete!")

if __name__ == "__main__":
    main()
