# casusal_energy_demand_prediction

This project is inspired by my diploma thesis titled "Causality Relationships and Prediction of Energy Consumption with Multivariate Machine Learning Models".
The purpose of this thesis is educational and is focused on the examination of the effect of the selection of parameters based on causality relationships in the prediction of electricity demand. It should be noted that the purpose of this thesis was not to find the best model, but to examine the causal relashionships of the variables in order to build an efficient and transparent solution. In this repository, parts of the overall work are uploaded, so as to demonstrate the main pipeline of the above research. 

First of all, the data preperation and visualization is covered. Moreover, the univariate linear and non-linear models will be implemented. The only timeseries that will be used for the construction and training of these models is the energy demand. Afterwards, two more models for each of them will be tested. In the so called "Full" model, all the available variables will be added to the models. Finally, the "Restricted" models will only use the variables that got selected by the Conditional Granger Causality Index (CGCI). All the linear and non-linear predictive models are examined and their performance is compared when only the driver variable is used, when all available variables are used and when only some of them are used. This process led to the conclusion that targeted variable selection based on the CGCI had a positive impact, as predictions were equivalently accurate to the full models, while there was a large reduction in training time.

Files and brief explanation:

- exploratory_data_analysis.ipynb: Jupyter Notebook that containts the data visualization and first look.
- order_selection_ar.ipynb: Jupyter Notebook that contains the order selection for the linear models.
- single_sar.ipynb: Jupyter Notebook that containts the  single variable linear model implementation.
- restricted_var.ipynb: Jupyter Notebook that containts the  restricted linear model implementation. That means that appart from energy consumption, other variables are also added, according to the direct granger index. 
- full_linear_var.ipynb: Jupyter Notebook that containts the  full linear model implementation. That means that all the available variables are added to the model. 
- LSTM_single.py: Python script that contains the single variable LSTM model implmentation.
- LSTM_restricted.py: Python script that contains the restricted LSTM model implmentation. That means that appart from energy consumption, other variables are also added, according to the direct granger index. 
- LSTM_full.py:  Python script that contains the full LSTM model implmentation. That means that all the available variables are added to the model. 
- functions.py: a script that contains some functions that help along the project pipeline and are used by multiple other scripts. 
