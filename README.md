# UPV FAQ Semantic Text Similarity Test

Only supports fastText embeddings in .bin format

## Dependencies:

- [fasttext](https://fasttext.cc/docs/en/python-module.html)
- [pandas](https://pandas.pydata.org)
- [numpy](https://numpy.org/install/)
- [scikit-learn](https://scikit-learn.org/stable/install.html)
- [seaborn](https://seaborn.pydata.org/installing.html)
- [matplotlib](https://matplotlib.org)
- [compress-fasttext](https://github.com/avidale/compress-fasttext) (optional)

## Usage:

    python3 upv50_tests.py model_path [--probs word_probs_path] [--alpha alpha] [--cmtime disp_time_seconds] [--cm] [--compressed] [--verb] [--save]
  
### Arguments:

- **model_path**: Path to the model to be evaluated

- **probs**: Path to vocabulary world frequencies - json file created using faq50.extract_word_probs

- **alpha**: Word embedding weighting parameter - word embeddings will be weighted based on word frequencies, according to this [paper](https://openreview.net/pdf?id=SyK00v5xx)

- **cmtime**: Cofusion matrix display duration in seconds

- **cm**: Show confusion matrices during evaluation

- **compressed**: Indicates usage of a compressed model created via [compress-fasttext](https://github.com/avidale/compress-fasttext)
  
- **verb**: Print incorrectly matched pairs in the following format:

    > Querry question : Incorrectly matched question (answer)

- **save**: Save evaluation results, including optional confusion matrices, into an appropriately named 
folder next to the evaluated model

## Confusion inspector:

    python3 upv50_confusion_inspector.py model_path [--probs word_probs_path] [--alpha alpha] [--compressed]

Shows a similarity heatmap between all dataset questions. Click on a specific pixel in the heatmap to print 
the corresponding pair of questions and their similarity value in the terminal.
