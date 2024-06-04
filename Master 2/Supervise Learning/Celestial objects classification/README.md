# Celestial Object Classification
Multiclass classification of celestial object. *November 2023*

## Context
Project that focuses on multiclass classification as part of the *Avanced Supervised Learning and Datachallenge* course within the M2 Mathematics and Artificial Intelligence program at the Institut Mathématique d'Orsay (IMO), Paris-Saclay University, under the supervision of Mr Olivier COUDRAY.  

Specifically, there exists a Kaggle competition for the class.

## Data
The project utilizes data sourced from https://www.sdss4.org/science/image-gallery/. 
Specifically, there are 8 available features used to predict the classification into one of three possible classes.

## Methodology
We begin our analysis by examining the data representation, which enables us to establish a preprocessing strategy aimed at reducing correlations among certain covariates. Subsequently, we systematically explore a wide range of supervised classification algorithms, beginning with basic linear methods and progressing to more sophisticated techniques such as Adaboost and neural networks. To enhance the robustness of our results, we employ various methods of aggregation.

Our ultimate goal is to enhance predictions for a test set in a Kaggle competition. To achieve this, we adopt a transductive perspective and leverage the inherent structure of the test set through a semi-supervised approach which is then modified to be more *robust*.

## How to run the notebook
The `numpy`, `pandas`, `matplotlib`, and `seaborn` libraries are required to run the notebook as well as `scikit-learn`, `xgboost` and `catboost` to import models.
Each package can be installed with the command line `pip install <package name>` or `conda install <package name>`. Once you have installed everything and you have uploaded the train and test files (named as such in the data folder), all you have to do is run the notebook.

## Further information
More details about the methodology followed are available in the report.

## Authors
*Lucas BIECHY* (lucas.biechy@university-paris-saclay.fr) and *Angel REYERO* (angel.reyero-lobo@universite-paris-saclay.fr).  

