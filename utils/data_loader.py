import numpy as np
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_processed_data(return_dataframe=False, max_jobs=None, random_seed=42):
    """
    Load pre-processed data from npz files
    
    Args:
        return_dataframe: If True, returns pandas DataFrames. If False, returns raw texts.
        max_jobs: Maximum number of jobs to load. If None, loads all jobs.
        random_seed: Seed for random sampling of jobs. Default is 42 for reproducibility.
        
    Returns:
        If return_dataframe=True:
            resumes_df: DataFrame with ID and processed_text columns
            jobs_df: DataFrame with job_id and combined_text columns
        If return_dataframe=False:
            resume_texts: Array of resume texts
            job_texts: Array of job texts
    """
    logger.info("Loading pre-processed data...")
    
    # Load resume data
    resume_data = np.load('./data/clean/resume_data.npz', allow_pickle=True)
    resume_ids = resume_data['ids']
    resume_texts = resume_data['texts']
    
    # Load job data
    job_data = np.load('./data/clean/job_data.npz', allow_pickle=True)
    job_ids = job_data['ids']
    job_texts = job_data['texts']
    
    # Limit number of jobs if specified
    if max_jobs is not None:
        logger.info(f"Randomly selecting {max_jobs} jobs")
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        # Generate random indices
        random_indices = np.random.choice(len(job_ids), size=max_jobs, replace=False)
        # Select jobs using random indices
        job_ids = job_ids[random_indices]
        job_texts = job_texts[random_indices]
    
    if return_dataframe:
        # Convert to DataFrames for consistency
        resumes_df = pd.DataFrame({
            'ID': resume_ids,
            'processed_text': resume_texts
        })
        
        jobs_df = pd.DataFrame({
            'job_id': job_ids,
            'combined_text': job_texts
        })
        
        return resumes_df, jobs_df
    else:
        # Convert numpy arrays to lists for text data
        return resume_texts.tolist(), job_texts.tolist() 