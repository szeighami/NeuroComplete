## Instalation and requirements
The prgram requires tensorflow, numpy, pandas and sklearn. It has been tested with tesorflow 2.4.0, numpy 1.19.5, pandas 1.0.1 and sklearn 0.22.1.

## Running NeuroComplete
Running NeuroComplete requires first craeting a config file containing all the configurations required and then calling python query.py to train the model and perform inference. The program automatically generates training and testing data for the H1 setting on a subsample of housing dataset, trains NeuroComplete on the training data. To reproduce experiments, first download and preprocess data using [ReStore](https://github.com/DataManagementLab/restore), place them in the data folder and call python query.py


## Config file
The file default\_config.py contains the default configuration and explanation for each parameter. Running python default\_config.py creates a json file containing default values.
