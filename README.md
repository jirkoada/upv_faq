# UPV FAQ Semantic Text Similarity Test

Only supports fastText embeddings in .bin format

## Dependecies:

- [fasttext](https://fasttext.cc/docs/en/python-module.html)
- [numpy](https://numpy.org/install/)
- [scikit-learn](https://scikit-learn.org/stable/install.html)
- [seaborn](https://seaborn.pydata.org/installing.html)

## Usage:

    python3 upv_tests.py [--verb] [--cm] [--save] model_path
  
### Arguments:

- **model_path**: Path to the model to be evaluated

- **verb**: Print incorrectly matched pairs in the following format:

    > Querry question : Incorrectly matched question (answer)

- **cm**: Show confusion matrices during evaluation

- **save**: Save evaluation results, including optional confusion matrices, into an appropriately named folder next to the evaluated model
