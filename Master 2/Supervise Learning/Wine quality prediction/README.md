# Wine quality prediction
Wine quality prediction through regression methods. *November 2023*

## Context
Project that focuses on regression as part of the *Avanced Supervised Learning and Datachallenge* course within the M2 Mathematics and Artificial Intelligence program at the Institut Mathématique d'Orsay (IMO), Paris-Saclay University, under the supervision of Mr Olivier COUDRAY.

Specifically, there exists a Kaggle competition for the class.

## Data
The project utilizes data sourced from https://www.sdss4.org/science/image-gallery/.

There are 12 available features, all continuous, except for the wine type, which is represented as 0 for red wine and 1 for white wine. These features are used to predict the quality of the wine on a scale from 1 to 10.

## Methodology
We begin the analysis by examining the data representation, enabling us to partition the data into categories of white and red wine, facilitating the creation of distinct models for each type. Subsequently, we explore a diverse range of supervised regression algorithms, starting with standard linear regression and advancing to more intricate techniques like GAMs, SVR, and Boosting. Additionally, we implement various aggregation methods in our approach.

## How to run the notebook
The `numpy`, `pandas`, `matplotlib`, `mmdfuse`, `jax`, `scipy` and `seaborn` libraries are required to run the notebook as well as `scikit-learn`, `pygam`, `xgboost` and `catboost` to import models.
Each package can be installed with the command line `pip install <package name>` or `conda install <package name>` but `mmdfuse` and `jax`.
The `mmdfuse` library (based on the jax library) is a few months old and is available at https://github.com/antoninschrab/mmdfuse/commits/master. To install them, run the `pip install git+https://github.com/antoninschrab/mmdfuse.git command`. You may need to add the kernel function (found here: https://github.com/antoninschrab/mmdfuse-paper/blob/master/kernel.py) in a python script in the same folder as the notebook to ensure it works properly.
Once you have installed everything and you have uploaded the train and test files (named as such in the data folder), all you have to do is run the notebook.

## Further information
More details about the methodology followed are available in the report.

## Authors
*Lucas BIECHY* (lucas.biechy@university-paris-saclay.fr) and *Angel REYERO* (angel.reyero-lobo@universite-paris-saclay.fr).  
