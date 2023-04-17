# Next-Word-Wizard-Group-18-NLP-Project-
Next Word Wizard (Group 18 NLP Project)


Datasets for All Models: https://drive.google.com/drive/folders/1uCNCton7PudpfyZGcLxZDRc0OxfbwF6c?usp=sharing

Saved Models for LSTM, BiLSTM, and Grid Search CV for Ablation Study: https://drive.google.com/drive/folders/1TbEyO0LpIiUDS-dDdFTQOqrJzwL4hofk?usp=sharing


Instructions to run the code for the different Models:

1. For Markov Model: 
    - Run the cells in the 'Markov Chain Model.ipynb' notebook. The dataset used in this notebook is 'merged-articles.txt'
    - Enter an input sequence of choice in the last cell of the notebook. Upon running this cell, the next word predictions are generated as the output of this cell.
    
2. For LSTM, BiLSTM, and Grid Search CV: 
    - Follow the steps in the 'NLP_Project_Group_18_LSTM_BiLSTM.ipynb' notebook. 
    - The dataset used in this notebook is 'ArticlesApril2017.csv'
    - Upon loading the file, follow the steps for Text Pre-processing and Tokenization.
    - Load the 100 dimensional GloVe embeddings using 'glove.6B.100d.txt'
    - Run the cells for the TSNE visualizations of the GloVe embeddings of the Top 10 Most frequent words.
    - Load the models from the above Google Drive Link using 'pickle' library.
   
3. For Transformer Models: 
    - Follow the steps in the 'Next word Prediction using various Transformer Models.ipynb' notebook.
    - The dataset used in this notebook for training is 'merged_articles.txt' 
    - Install the Happy Transformer Library and train it on merged_articles.txt' dataset.
    - Run the cells for training of the four Transformer models and enter a sentence to get the Top 10 probable words along with the probabilities.
    - Save the models for performing inference.
    

4. To load the Streamlit Application, first save the trained model weights and then run the following command.

``` 
streamlit run main.py

```

This loads the Web application to perform inference using various models.
