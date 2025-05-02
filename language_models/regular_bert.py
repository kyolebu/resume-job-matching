import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

from utils.data_loader import load_processed_data
from utils.matching import compute_similarity, find_top_matches, save_matches

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RegularBERTEmbedder:
    def __init__(self, model_name='bert-base-uncased', max_length=512):
        """
        Initialize BERT model and tokenizer
        
        Args:
            model_name: Name of the BERT model to use
            max_length: Maximum sequence length for tokenization
        """
        logger.info(f"Loading BERT model: {model_name}")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model loaded and moved to {self.device}")

    def get_embeddings(self, texts, batch_size=32):
        """
        Generate BERT embeddings for a list of texts
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            numpy array of embeddings
        """
        logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size)):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                # Move to device
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                # Get embeddings
                outputs = self.model(input_ids, attention_mask=attention_mask)
                
                # Use [CLS] token embedding as the sentence embedding
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)

def main(max_jobs=None):
    # Initialize BERT embedder
    embedder = RegularBERTEmbedder()
    
    # Load pre-processed data
    resume_texts, job_texts = load_processed_data(return_dataframe=False, max_jobs=max_jobs)
    
    # Generate embeddings
    logger.info("Generating resume embeddings...")
    resume_embeddings = embedder.get_embeddings(resume_texts)
    
    logger.info("Generating job embeddings...")
    job_embeddings = embedder.get_embeddings(job_texts)
    
    # Save embeddings
    logger.info("Saving embeddings...")
    np.savez('./data/embeddings/regular_bert_resume_embeddings.npz', embeddings=resume_embeddings)
    np.savez('./data/embeddings/regular_bert_job_embeddings.npz', embeddings=job_embeddings)
    
    # Compute similarities
    similarity = compute_similarity(resume_embeddings, job_embeddings)
    
    # Find top matches
    top_indices, top_scores = find_top_matches(similarity)
    
    # Save matches
    save_matches(top_indices, top_scores, './data/matches/regular_bert_matches.json')
    
    # Print example matches
    logger.info("\nExample matches for first resume:")
    for i, (job_idx, score) in enumerate(zip(top_indices[0], top_scores[0])):
        logger.info(f"Match {i+1}: Job {job_idx} (similarity: {score:.4f})")
    
    logger.info("Processing complete!")

if __name__ == "__main__":
    # Process 5000 jobs
    main(max_jobs=5000) 