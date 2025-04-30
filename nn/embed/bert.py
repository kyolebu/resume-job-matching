import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BERTEmbedder:
    def __init__(self, model_name='all-MiniLM-L6-v2', batch_size=16):
        """
        Initialize Sentence-BERT embedder.

        Args:
            model_name (str): Name of the sentence-transformers model to use.
            batch_size (int): Batch size for processing.
        """
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size

    def get_embeddings(self, texts, chunk_size=1000):
        """
        Get sentence embeddings for a list of texts, processing in chunks.

        Args:
            texts (list): List of text strings to embed.
            chunk_size (int): Number of texts to process at once.

        Returns:
            numpy.ndarray: Array of embeddings.
        """
        all_embeddings = []
        for chunk_start in tqdm(range(0, len(texts), chunk_size)):
            chunk = texts[chunk_start:chunk_start + chunk_size]
            embeddings = self.model.encode(
                chunk, 
                batch_size=self.batch_size, 
                show_progress_bar=False, 
                convert_to_numpy=True
            )
            all_embeddings.extend(embeddings)

            # Save intermediate results
            #if chunk_start + chunk_size < len(texts):
            #    np.save(f'./data/embeddings/temp_embeddings_{chunk_start}.npy', np.array(all_embeddings))

        return np.array(all_embeddings)

def process_data(input_dir='./data/clean', output_dir='./data/embeddings'):
    """
    Process resumes and job postings to generate BERT embeddings
    
    Args:
        input_dir (str): Directory containing input CSV files
        output_dir (str): Directory to save embeddings
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading data...")
    resume_df = pd.read_csv(f'{input_dir}/processed_resumes.csv')
    job_df = pd.read_csv(f'{input_dir}/processed_jobs.csv')
    
    # Clean data - ensure all text is string type and not NaN
    resume_df['processed_text'] = resume_df['processed_text'].fillna('').astype(str)
    job_df['combined_text'] = job_df['combined_text'].fillna('').astype(str)
    
    # Initialize BERT embedder with specified model
    embedder = BERTEmbedder(model_name='all-MiniLM-L6-v2', batch_size=16)
    
    # Process resumes
    logger.info("Processing resumes...")
    resume_embeddings = embedder.get_embeddings(resume_df['processed_text'].tolist())
    
    # Save resume embeddings
    logger.info("Saving resume embeddings...")
    np.save(f'{output_dir}/resume_embeddings.npy', resume_embeddings)
    resume_df['ID'].to_csv(f'{output_dir}/resume_ids.csv', index=False)
    
    # Process jobs in chunks
    logger.info("Processing jobs...")
    job_embeddings = embedder.get_embeddings(job_df['combined_text'].tolist(), chunk_size=1000)
    
    # Save job embeddings
    logger.info("Saving job embeddings...")
    np.savez_compressed(f'{output_dir}/job_embeddings.npz', job_embeddings=job_embeddings)
    job_df['job_id'].to_csv(f'{output_dir}/job_ids.csv', index=False)
    
    logger.info("Processing complete!")

if __name__ == "__main__":
    process_data()
