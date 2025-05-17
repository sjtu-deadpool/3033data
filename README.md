# Multi-modal Movie Recommendation System running guide

## Project Overview

This project is a multi-modal movie recommendation system that integrates visual features, textual features to predict user ratings for movies. The system creates a comprehensive representation of movies by fusing different types of features, enabling more accurate capture of movie content and user preferences.

## Prerequisites

- Python 3.8+
- PyTorch 1.9+
- pandas, numpy, scikit-learn
- matplotlib, seaborn
- requests, BeautifulSoup4
- transformers (Hugging Face)
- tqdm
- dataset link: https://drive.google.com/drive/u/0/folders/1HJVO6_PCEX7AIyPygX3d9X4-rF6lnZDV

## API Keys Setup

Before running the scripts, you'll need to set up the following API keys:

1. **GROK API Key (Recommended)**
   - Used for enriching movie plots and tagline explanations
   - Why use GROK? It offers substantial free quotas for text enrichment
   - Note: You can use your ChatGPT API key as a substitute

2. **SerpAPI Key (Optional)** 
   - Used for retrieving movie still frames and images
   - Alternative: Without this, `data_digestion.py` will directly obtain 5 still frames without filtering

3. **TMDb API Key (Recommended)**
   - Used for movie metadata retrieval
   - Free to register and use with generous quotas

Note: IMDb data is accessed via Cinemagoer, which doesn't require API key registration.

## Execution Order and Functionality

### 1. Data Collection and Processing (`data_digestion.py`)
- Collects basic information about IMDb Top 250 movies
- Retrieves titles, years, plots, directors, and taglines
- Can directly acquire 5 static frame images per movie

### 2. Movie Image Acquisition (`picture_acquisition.py`)
- Downloads movie posters and still frame images
- Creates organized image repository for feature extraction

### 3. Feature Selection (`resnet_select.py`)
- Optional processing for Picture features
- Can be skipped if using images directly acquired without filtering

### 4. Visual Feature Extraction (`Visual_feature_vit.py`)
- Processes movie images with Vision Transformer (ViT)
- Extracts 768-dimensional visual feature vectors
- Creates aggregated visual representation from multiple images

### 5. Textual Feature Extraction (`textual_feature.py`)
- Processes movie plots, trailer texts, and taglines with BERT
- Extracts 2304-dimensional textual feature vectors
- Captures semantic content and thematic elements

### 6. Data Merging and Filtering (`merge_data_250filterVersion.py`)
- Merges IMDb Top 250 movie data with metadata and ratings
- Applies filters to retain relevant movies
- Prepares data for feature fusion

### 7. Feature Fusion (`fusion_pipeline.py`)
- Combines visual features (768-D), textual features (2304-D), structured metadata (128-D), and genome tags (1128-D)
- Creates unified 4360-dimensional feature vectors
- Prepares data for model training

### 8. Model Training and Ablation Studies (`predict_model&ablation.py`)
- Trains MLP regressor for rating prediction
- Conducts three ablation studies:
  - Full model (RMSE: 0.410, MAE: 0.310)
  - Without visual features (RMSE: 0.434, MAE: 0.330)
  - Without genome tags (RMSE: 0.445, MAE: 0.339)
- Evaluates model performance with RMSE and MAE metrics

## Data Flow

1. **Input Data**: MovieLENSdataset, CinemaGoer Api(IMDB), TMDB, google serpapi api, Grok or other llm api
2. **Feature Extraction**: Visual, textual, structured, and genome features in MovieLens dataset
3. **Feature Fusion**: Concatenation into unified representations
4. **Rating Prediction**: MLP regressor for personalized recommendations

## References

- MovieLens dataset
- IMDb database
- Hugging Face Transformers library
- PyTorch deep learning framework