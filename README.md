# CS-412-Project
The efficient processing of long documents remains a significant challenge due to their large token size and the quadratic complexity of the attention mechanism in Large Language Models (LLMs). In this project, we investigate the ability of LLMs to handle long documents. While prior research was sceptical in LLMs effectiveness in this domain, we explore its capability along with the state-of-the-art embedding models and traditional machine learning techniques to evaluate their effectiveness on long documents. Tasks may span several NLP sub-fields and involve text generation and classification. By leveraging open-source language models, we aim to achieve and potentially exceed state-of-the-art performance in long-form document processing.

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
To run on Windows you might need to use Windows Subsystem for Linux as VLLM is not (as of writing) available for windows. Otherwise, you need to do the following:
    1. Request an access token to your Huggingface account (you need one for access to Llama) 
        (https://huggingface.co/docs/transformers/main/en/model_doc/llama) You need one for 3.1 and 3.2
    2. Save it in token.txt in the same folder as the code
    3. Need to be running python 3.10 to run vllm, need CUDA 12.4 (other versions might also work but not sure)
    4. run the following:
        pip install vllm
        pip install datasets
        pip install huggingface_hub
        pip install nltk
    5. You should be good to go (as long as you have a large enough GPU (we ran this on Nvidia A5000))

Hope it works! If not let us know we'll provide guidance.


### Nomic-Embed Classification

Unzip the train/validation/test datasets at `Nomic-Embed/Dataset/muld-data`

Run the python code in `Nomic-Embed` for Quality and muld-movie to evaluate the accuracy of the models.

### Traditional Machine-Learning
The machine learning & a deep learning pipeline designed for text classification on the MULD/Quality dataset. The pipeline utilizes various preprocessing techniques, feature extraction using Word2Vec embeddings, and an ensemble of classifiers/ neural networks for classification tasks.
