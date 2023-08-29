## Instalation and requirements
The prgram requires tensorflow, numpy, pandas and jax. It has been tested with tesorflow 2.4.0, numpy 1.19.5, pandas 1.0.1 and jax 0.3.14.

## Running NeuroComplete
Running NeuroComplete requires first creating a config file containing all the configurations required and then calling `python query.py` to train the model and perform inference. The program automatically generates training and testing data for the H1 setting on a subsample of housing dataset, trains NeuroComplete on the training data and performs inference for H1 average queries.

## Config file
The file `default_config.py` contains the default configuration and explanation for each parameter. Running `python default_config.py` creates a json file containing default values.
