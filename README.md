# UPV FAQ Semantic Text Similarity Test

Only supports fastText embeddings in .bin format

## Usage:

    python3 upv_tests.py [--verb] [--cm] [--save] model_path
  
### Arguments:

- **model_path**: Path to the model to be evaluated

- **verb**: Print incorrectly matched pairs in the following format:

    > Querry question : Incorrectly matched question (answer)

- **cm**: Show confusion matrices during evaluation

- **save**: Save evaluation results, including optional confusion matrices, into an adequately named subfolder next to the evaluated model
