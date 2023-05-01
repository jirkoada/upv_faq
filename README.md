# UPV FAQ Semantic Text Similarity Test

Only supports fastText embeddings in .bin format

## Dependencies:

- [fasttext](https://fasttext.cc/docs/en/python-module.html)
- [pandas](https://pandas.pydata.org)
- [numpy](https://numpy.org/install/)
- [scikit-learn](https://scikit-learn.org/stable/install.html)
- [seaborn](https://seaborn.pydata.org/installing.html)
- [matplotlib](https://matplotlib.org)

## Usage:

    python3 upv50_tests.py [--verb] [--cm] [--save] model_path
  
### Arguments:

- **model_path**: Path to the model to be evaluated

- **verb**: Print incorrectly matched pairs in the following format:

    > Querry question : Incorrectly matched question (answer)

- **cm**: Show confusion matrices during evaluation

- **save**: Save evaluation results, including optional confusion matrices, into an appropriately named 
folder next to the evaluated model

## Confusion inspector:

    python3 upv50_confusion_inspector.py model_path

Shows a similarity heatmap between all dataset questions. Click on a specific pixel in the heatmap to print 
the corresponding pair of questions and their similarity value in the terminal.