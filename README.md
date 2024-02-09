# Twitter-Sentiment-Analysis

This repository contains Jupyter notebooks implementing various deep learning models for sentiment analysis on Twitter data. The models explored in the notebooks include BERT, CNN, LSTM, BiLSTM, and combinations like BERT-CNN, BERT-LSTM, and BERT-BiLSTM. The sentiment analysis task involves predicting the sentiment (positive, negative, neutral, or irrelevant) associated with Twitter entities.

## Dataset

The dataset used for this project can be found [here](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis). Before running the notebooks, please download and move the dataset files from the `data/` directory to the same directory as the Jupyter Notebooks files. The dataset description mentions irrelevant as a neutral category, but we have considered it a separate class label.

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

This notebook covers the training and validation results of CNN, LSTM, and BiLSTM models for sentiment analysis. Due to the large sizes of machine learning models, the models are not included in the git repository.

## Note:

The notebooks where the models have been tested have used models saved during training. Due to size limitations in GitHub the models are not uploaded. You may run the training notebooks to generate and save the models and use those models in the testing code.

## Usage

1. **Dataset Preparation:** Download the dataset from the provided link and place it in the same directory as the Jupyter Notebook files.
2. **Jupyter Notebooks:** Open and execute the desired notebook(s) for exploring the models and performing sentiment analysis tasks.
3. **Model Experimentation:** Feel free to experiment, modify, or combine the models within the notebooks to improve sentiment analysis performance.

## Results and Evaluation

Our comprehensive analysis across various models unveiled distinct performance metrics, highlighting the nuanced capabilities of each architecture in sentiment analysis on Twitter data. BERT-based models, due to their advanced understanding of context, outperformed traditional CNN, LSTM, and BiLSTM models. Specifically, the fusion models such as BERT-CNN, BERT-LSTM, and BERT-BiLSTM demonstrated superior accuracy, leveraging both the contextual awareness of BERT and the unique strengths of CNNs and LSTMs in text classification.

For a more detailed comparison and analysis of the performance metrics, including accuracy, precision, recall, and F1 scores of each model, we encourage readers to refer to the individual Jupyter notebooks. These notebooks provide a comprehensive evaluation framework, including confusion matrices and classification reports, to support a deeper understanding of each model's effectiveness in sentiment classification.

This section encapsulates the essence of our project's evaluation phase, providing a snapshot of our findings. For a granular view and technical details, diving into the notebooks is highly recommended.

## Requirements

- Jupyter Notebook
- Python (>=3.6)
- TensorFlow or PyTorch
- Hugging Face Transformers library
- NumPy, Pandas, etc. (common Python libraries)

## Contributors

- Ankith Reddy Avula [@avulaankith](https://github.com/avulaankith)
- Prem Kumar Rohan
- Manogna Shashidhara

## License

This project is licensed under the [Apache License 2.0](LICENSE).
