# cs760-project
A repository to store code related to the CS760 ML Final Project that Jason, Samantha, and I are working on.

## Sub-Directories
1. DecisionTreeRegression
    * A Mathematica Notebook containing an implementation of a binary decision tree with linear regression to predict COVID severity indices for counties in the United States.

2. KNearestNeighbors
    *

3. LinearRegression
   * A matlab program implementing Linear Regression on the data set produced from the RawDataToUsableDataConverter. This Linear Regression implemention is explored in a homoskedastic model as well as a heteroskedastic model so the team can understand how the different assumptions change the outcome in our case. Some of the misc. files in this directory are needed for matlab's integration with Git, so please ignore them and do not remove them.

4. NeuralNetwork
    * A python program that trains a shallow fully connected feed-forward neural network as a regression predictor. Implemented using Tensorflow and the Adaptive Moment (Adam) Optimizer with lasso regularization and mean-squared error as cost metrics.

5. RawDataToUsableDataConverter
    * A java program that takes in the raw data files we want to analyze and outputs an aggregate data file
that is properly formatted, contains only the features we are interested in, maps the time series of individual county COVID data
to a final severity index, and is ready to use in further ML analysis

6. SeverityIndexVisualization
    * A Mathematica Notebook containing code used for the visualizations of the COVID severity index found in Appendix B.
