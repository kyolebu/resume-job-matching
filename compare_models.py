import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import matplotlib.ticker as mticker
from language_models.regular_bert import RegularBERTEmbedder
from language_models.sentence_bert import generate_embeddings
from baseline.tfidf import compute_tfidf_matches

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for all plots
plt.style.use('default')
sns.set_style("whitegrid")
sns.set_palette("husl")

def load_matches():
    """Load matches from all models"""
    logger.info("Loading matches from all models...")
    
    # Load matches from JSON files
    with open('./data/matches/tfidf_matches.json', 'r') as f:
        tfidf_matches = json.load(f)
    
    with open('./data/matches/bert_matches.json', 'r') as f:
        sentence_bert_matches = json.load(f)
    
    with open('./data/matches/regular_bert_matches.json', 'r') as f:
        regular_bert_matches = json.load(f)
    
    return tfidf_matches, sentence_bert_matches, regular_bert_matches

def compute_overlap_metrics(bert_matches, tfidf_matches, regular_bert_matches):
    """Compute overlap between all three models"""
    logger.info("Computing overlap metrics...")
    
    overlap_stats = {
        'bert_tfidf_overlap': [],
        'bert_regular_overlap': [],
        'tfidf_regular_overlap': [],
        'all_models_overlap': [],
        'similarity_diffs': {
            'bert_tfidf': [],
            'bert_regular': [],
            'tfidf_regular': []
        },
        'scores': {
            'bert': [],
            'tfidf': [],
            'regular_bert': []
        },
        'position_overlaps': {
            'bert_tfidf': defaultdict(int),
            'bert_regular': defaultdict(int),
            'tfidf_regular': defaultdict(int)
        }
    }
    
    for b_match, t_match, r_match in zip(bert_matches, tfidf_matches, regular_bert_matches):
        # Get job indices for all models
        bert_jobs = [m['job_index'] for m in b_match['top_matches']]
        tfidf_jobs = [m['job_index'] for m in t_match['top_matches']]
        regular_jobs = [m['job_index'] for m in r_match['top_matches']]
        
        # Calculate pairwise overlaps
        overlap_stats['bert_tfidf_overlap'].append(len(set(bert_jobs) & set(tfidf_jobs)))
        overlap_stats['bert_regular_overlap'].append(len(set(bert_jobs) & set(regular_jobs)))
        overlap_stats['tfidf_regular_overlap'].append(len(set(tfidf_jobs) & set(regular_jobs)))
        overlap_stats['all_models_overlap'].append(len(set(bert_jobs) & set(tfidf_jobs) & set(regular_jobs)))
        
        # Calculate position-wise overlaps
        for pos in range(5):
            if bert_jobs[pos] == tfidf_jobs[pos]:
                overlap_stats['position_overlaps']['bert_tfidf'][pos] += 1
            if bert_jobs[pos] == regular_jobs[pos]:
                overlap_stats['position_overlaps']['bert_regular'][pos] += 1
            if tfidf_jobs[pos] == regular_jobs[pos]:
                overlap_stats['position_overlaps']['tfidf_regular'][pos] += 1
        
        # Get similarity scores
        bert_scores = [m['similarity_score'] for m in b_match['top_matches']]
        tfidf_scores = [m['similarity_score'] for m in t_match['top_matches']]
        regular_scores = [m['similarity_score'] for m in r_match['top_matches']]
        
        # Store scores
        overlap_stats['scores']['bert'].extend(bert_scores)
        overlap_stats['scores']['tfidf'].extend(tfidf_scores)
        overlap_stats['scores']['regular_bert'].extend(regular_scores)
        
        # Calculate average score differences
        overlap_stats['similarity_diffs']['bert_tfidf'].append(
            np.mean(np.abs(np.array(bert_scores) - np.array(tfidf_scores)))
        )
        overlap_stats['similarity_diffs']['bert_regular'].append(
            np.mean(np.abs(np.array(bert_scores) - np.array(regular_scores)))
        )
        overlap_stats['similarity_diffs']['tfidf_regular'].append(
            np.mean(np.abs(np.array(tfidf_scores) - np.array(regular_scores)))
        )
    
    return overlap_stats

def plot_comparisons(overlap_stats, output_dir='./data/matches/plots'):
    """Generate comparison plots"""
    logger.info("Generating comparison plots...")
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Overlap Distribution Plot
    plt.figure(figsize=(15, 6))
    overlap_data = pd.DataFrame({
        'BERT-TFIDF': overlap_stats['bert_tfidf_overlap'],
        'BERT-Regular': overlap_stats['bert_regular_overlap'],
        'TFIDF-Regular': overlap_stats['tfidf_regular_overlap']
    })
    sns.boxplot(data=overlap_data)
    plt.title('Distribution of Overlaps between Model Pairs', pad=20)
    plt.ylabel('Number of Overlapping Matches')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/overlap_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Score Distribution Comparison
    plt.figure(figsize=(15, 6))
    for model, scores in overlap_stats['scores'].items():
        sns.kdeplot(scores, label=model)
    plt.title('Distribution of Similarity Scores', pad=20)
    plt.xlabel('Similarity Score')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/score_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Position-wise Overlap
    plt.figure(figsize=(15, 6))
    positions = list(range(5))
    width = 0.25
    
    for i, (pair, overlaps) in enumerate(overlap_stats['position_overlaps'].items()):
        plt.bar([p + i*width for p in positions], 
                [overlaps[p] for p in positions], 
                width, label=pair)
    
    plt.title('Position-wise Agreement between Models', pad=20)
    plt.xlabel('Position in Top-5')
    plt.ylabel('Number of Matches')
    plt.xticks([p + width for p in positions], [f'Position {p+1}' for p in positions])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/position_agreement.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Score Difference Distribution
    plt.figure(figsize=(15, 6))
    for pair, diffs in overlap_stats['similarity_diffs'].items():
        sns.kdeplot(diffs, label=pair)
    plt.title('Distribution of Average Score Differences', pad=20)
    plt.xlabel('Average Score Difference')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/score_differences.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_results(overlap_stats):
    """Analyze and summarize comparison results"""
    logger.info("Analyzing results...")
    
    results = {
        'average_overlaps': {
            'bert_tfidf': np.mean(overlap_stats['bert_tfidf_overlap']),
            'bert_regular': np.mean(overlap_stats['bert_regular_overlap']),
            'tfidf_regular': np.mean(overlap_stats['tfidf_regular_overlap']),
            'all_models': np.mean(overlap_stats['all_models_overlap'])
        },
        'score_stats': {
            'bert': {
                'mean': np.mean(overlap_stats['scores']['bert']),
                'std': np.std(overlap_stats['scores']['bert']),
                'min': np.min(overlap_stats['scores']['bert']),
                'max': np.max(overlap_stats['scores']['bert'])
            },
            'tfidf': {
                'mean': np.mean(overlap_stats['scores']['tfidf']),
                'std': np.std(overlap_stats['scores']['tfidf']),
                'min': np.min(overlap_stats['scores']['tfidf']),
                'max': np.max(overlap_stats['scores']['tfidf'])
            },
            'regular_bert': {
                'mean': np.mean(overlap_stats['scores']['regular_bert']),
                'std': np.std(overlap_stats['scores']['regular_bert']),
                'min': np.min(overlap_stats['scores']['regular_bert']),
                'max': np.max(overlap_stats['scores']['regular_bert'])
            }
        },
        'similarity_differences': {
            'bert_tfidf': np.mean(overlap_stats['similarity_diffs']['bert_tfidf']),
            'bert_regular': np.mean(overlap_stats['similarity_diffs']['bert_regular']),
            'tfidf_regular': np.mean(overlap_stats['similarity_diffs']['tfidf_regular'])
        }
    }
    
    return results

def print_summary(results):
    """Print a human-readable summary of the results"""
    logger.info("\nModel Comparison Summary:")
    logger.info("-" * 50)
    
    logger.info("\nAverage Overlaps:")
    for pair, avg in results['average_overlaps'].items():
        logger.info(f"{pair}: {avg:.2f} out of 5")
    
    logger.info("\nScore Statistics:")
    for model, stats in results['score_stats'].items():
        logger.info(f"\n{model.upper()}:")
        logger.info(f"  Mean: {stats['mean']:.4f}")
        logger.info(f"  Std: {stats['std']:.4f}")
        logger.info(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    
    logger.info("\nAverage Similarity Differences:")
    for pair, diff in results['similarity_differences'].items():
        logger.info(f"{pair}: {diff:.4f}")

def save_results(results, output_file='./data/matches/all_models_comparison.json'):
    """Save comparison results to JSON file"""
    logger.info(f"Saving results to {output_file}...")
    
    # Create output directory if it doesn't exist
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("Results saved successfully!")

def plot_model_comparison_scatter(overlap_stats, output_dir='./data/embeddings/plots'):
    """Create scatter plot comparing TF-IDF and regular BERT similarity scores"""
    logger.info("Generating model comparison scatter plot...")
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get scores
    tfidf_scores = np.array(overlap_stats['scores']['tfidf'])
    regular_bert_scores = np.array(overlap_stats['scores']['regular_bert'])
    
    # Create scatter plot
    plt.figure(figsize=(12, 10))
    plt.scatter(regular_bert_scores, tfidf_scores, alpha=0.2, color='pink', s=20)
    
    # Add diagonal line
    plt.plot([0, 1], [0, 1], 'r--', alpha=0.5)
    
    # Set labels and title
    plt.xlabel('Regular BERT Similarity Score')
    plt.ylabel('TF-IDF Similarity Score')
    plt.title('BERT vs TF-IDF Similarity Scores')
    
    # Set axis limits
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # Add grid
    plt.grid(True, alpha=0.2)
    
    # Save plot
    plt.savefig(f'{output_dir}/tfidf_vs_regular_bert_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load matches from all models
    tfidf_matches, sentence_bert_matches, regular_bert_matches = load_matches()
    
    # Compute overlap metrics
    overlap_stats = compute_overlap_metrics(sentence_bert_matches, tfidf_matches, regular_bert_matches)
    
    # Analyze results
    results = analyze_results(overlap_stats)
    
    # Generate plots
    plot_comparisons(overlap_stats)
    plot_model_comparison_scatter(overlap_stats)
    
    # Save results
    save_results(results)
    
    # Print summary
    print_summary(results)

if __name__ == "__main__":
    main() 