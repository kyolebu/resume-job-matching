# Resume-Job Matching System

A comparison of different language models for matching resumes to job descriptions.

## Models Implemented

1. **TF-IDF (Baseline)**
   - Traditional bag-of-words approach
   - Fast and memory-efficient
   - Mean similarity score: 0.45

2. **Sentence BERT**
   - Using 'all-MiniLM-L6-v2' model
   - Optimized for semantic similarity
   - Mean similarity score: 0.70

3. **Regular BERT**
   - Using 'bert-base-uncased'
   - Higher but less discriminative scores
   - Mean similarity score: 0.89

## Project Structure

```
project/
├── baseline/
│   └── tfidf.py           # TF-IDF implementation
├── language_models/
│   ├── regular_bert.py    # BERT base implementation
│   └── sentence_bert.py   # Sentence BERT implementation
├── utils/
│   ├── data_loader.py     # Data loading utilities
│   └── matching.py        # Matching utilities
└── data/
    ├── clean/            # Processed text data
    ├── embeddings/       # Model embeddings
    └── matches/          # Matching results
```

## Key Findings

- All three models show different similarity score distributions
- Regular BERT produces highest scores but with least variation
- TF-IDF shows widest score range but lowest mean
- Sentence BERT provides balanced score distribution

## Usage

```bash
# Run individual models
python baseline/tfidf.py
python language_models/sentence_bert.py
python language_models/regular_bert.py

# Compare all models
python compare_models.py
```

