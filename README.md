# Twitter-Sentiment-Analysis

This repository contains Jupyter notebooks implementing various deep learning models for sentiment analysis on Twitter data. The models explored in the notebooks include BERT, CNN, LSTM, BiLSTM, and combinations like BERT-CNN, BERT-LSTM, and BERT-BiLSTM. The sentiment analysis task involves predicting the sentiment (positive, negative, or neutral) associated with Twitter entities.

## Dataset

The dataset used for this project can be found [here](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis). Please download and move the dataset files from the `data/` directory to the same directory as the jupyter notebooks files before running the notebooks.

## Notebooks

### 1. BERT_CNN.ipynb

This notebook implements the BERT-CNN model for sentiment analysis using Twitter data.

### 2. BERT_BILSTM.ipynb

The BERT-BiLSTM model is implemented in this notebook to perform sentiment analysis on Twitter entities.

### 3. BERT_final.ipynb

An all-encompassing notebook showcasing the final implementation and analysis using the BERT model for sentiment classification.

### 4. BERT_LSTM.ipynb

This notebook focuses on utilizing BERT embeddings with LSTM architecture for sentiment analysis on Twitter data.

### 5. BERT_MODELS_testing_code.ipynb

Contains testing code and experimentation with various BERT-based models for sentiment analysis.

### 6. CNN_LSTM_BiLSTM testing code.ipynb

A notebook dedicated to testing and experimenting with CNN, LSTM, and BiLSTM models for sentiment analysis.

### 7. BERT.ipynb

An exploration of the BERT model and its application to sentiment analysis tasks using Twitter data.

### 8. CNN_LSTM_BILSTM_Trainval_results.ipynb

This notebook covers the training and validation results of CNN, LSTM, and BiLSTM models for sentiment analysis.

## Usage

1. **Dataset Preparation:** Download the dataset from the provided link and place it in the same directory as the jupyter notebook files.
2. **Jupyter Notebooks:** Open and execute the desired notebook(s) for exploring the models and performing sentiment analysis tasks.
3. **Model Experimentation:** Feel free to experiment, modify, or combine the models within the notebooks to improve sentiment analysis performance.

## Requirements

- Jupyter Notebook
- Python (>=3.6)
- TensorFlow or PyTorch
- Hugging Face Transformers library
- NumPy, Pandas, etc. (common Python libraries)

## Contributors

- [Avula Ankith](https://github.com/avulaankith) (@avulaankith)
- Prem Kumar Rohan
- Manogna Shashidhara

## License

This project is licensed under the [Apache License 2.0](LICENSE).
