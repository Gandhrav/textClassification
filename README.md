BBC News Classification using BERT

This GitHub repository contains code for text classification using BERT (Bidirectional Encoder Representations from Transformers) on the BBC News Classification dataset from Kaggle. BERT is a powerful pre-trained language model developed by Google that has achieved state-of-the-art results in various natural language processing (NLP) tasks, including text classification.
Dataset

The dataset used in this project is the BBC dataset, which contains approximately 2,126 news articles from the BBC across five different categories: business, entertainment, politics, sport, and tech. The dataset is available on Kaggle.
Dependencies

The following Python libraries are used in this project and are required to run the code:

    tensorflow
    keras
    pandas
    numpy
    matplotlib
    sklearn
    transformers (for BERT)

You can install these libraries using pip or conda before running the code.
Usage

    Clone the repository to your local machine.
    Install the required dependencies using pip or conda.
    Download the BBC dataset from Kaggle.
    Run the Text_Classification_using_BERT.ipynb Jupyter notebook to train and evaluate the BERT model on the dataset.
    You can modify the hyperparameters and other configurations in the notebook to experiment with different settings.
    The trained BERT model will be saved in the models folder, and the evaluation results will be displayed in the notebook.

Results

The notebook provides the accuracy for the trained BERT model on the BBC dataset. You can use these metrics to evaluate the performance of the model.
Credits

    The BERT model implementation in this project is based on the Hugging Face Transformers library, which provides pre-trained BERT models and tools for fine-tuning them on custom tasks.
    The BBC dataset used in this project is obtained from Kaggle, and credits go to the original data source: BBC News [https://www.kaggle.com/datasets/sainijagjit/bbc-dataset].

Feel free to use and modify the code as per your requirements.
Happy coding!