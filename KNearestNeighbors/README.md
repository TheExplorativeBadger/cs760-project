# K-Nearest Neighbors

A Matlab script containing an implementation of Euclidean distance weighted K-Nearest Neighbors to predict COVID severity indices for counties in the United States. 
Currently uses 10 iterations of 10-Fold Cross Validation; however this number can be easily changed via a variable within the script if you wish to use a different number of iterations or folds.

##Steps:

     1. Use the 'RawDataToUsableDataConverter' to generate 'FinalCombinedFeaturesWithSeverity.csv'
     2. Add the generated FinalCombinedFeaturesWithSeverity.csv file to the KNearestNeighbors sub-directory
     3. Run the KNearestNeighbors.m Matlab Script to generate results