# UVP FAQ Semantic Similarity Test

Only supports fastText embedding models in .bin formats

## Usage:

    python3 upv_tests.py [--verb] [--cm] [--save] model_path
  
### Arguments:

model_path: Path to the model to be evaluated

--verb: If used, incorrectly matched pairs will be printed in the following format:

> Querry question : Incorrectly matched question (answer)

--cm: Show confusion matrices during evaluation

--save: Save evaluations results, including optional confusion matrices into an adequatly named subfolder next to the evaluated model
