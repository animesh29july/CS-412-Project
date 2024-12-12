# CS-412-Project
This project aims to thoroughly explore and evaluate methods for handling long documents in Natural Language Processing (NLP), focusing on their unique challenges, such as managing large input token counts without compromising coherence or computational efficiency. Tasks may span several NLP sub-fields and involve text generation and classification. By leveraging and contributing to open-source language models, we aim to achieve and potentially exceed state-of-the-art performance in long-form document processing. For this project we evaluate the prediction accuracies using the Large Language Models, state-of-the-art Nomic-Embedding model and traditional machine learning techniques. 

## Datasets

- Quality
    - Article: Contextual Input Text (String)
    - Question: Target Question (String)
    - Options: Possible answers to choose from (List)
    - Answers: Index of the correct answer in options list (Integer)
    - Hard: Measure of difficulty based on agreement of human annotators (Boolean)


- Muld-Movie
    - Input: A movie script and a character
    - Output: Classification of “hero” or “villain/antagonist”


### LLM

### Nomic-Embed Classification

Unzip the train/validation/test datasets at `Nomic-Embed/Dataset/muld-data`

Run the python code in `Nomic-Embed` for Quality and muld-movie to evaluate the accuracy of the models.

### Traditional Machine-Learning
The machine learning & a deep learning pipeline designed for text classification on the MULD/Quality dataset. The pipeline utilizes various preprocessing techniques, feature extraction using Word2Vec embeddings, and an ensemble of classifiers/ neural networks for classification tasks.
