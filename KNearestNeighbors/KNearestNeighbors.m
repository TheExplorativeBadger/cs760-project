%%% Problem Setup %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
filename = 'FinalCombinedFeaturesWithSeverity.csv';
OriginalFile = dlmread(filename, ';');

% y = covid severity index 
y = OriginalFile(:,1);
Dimensions_y = size(y);
% X = Demographic-Feature-Matrix
X = OriginalFile(:,2:end);
Dimensions_X = size(X);
Num_Rows_X = Dimensions_X(1);
Num_Columns_X = Dimensions_X(2);

%%% Find the Optimal K Value (Number Neighbors) Among a Specified Range %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
StartingNumNeighbors = 1;
EndingNumNeighbors = 25;
NumberFolds = 10;

NumberNeighbors = [];
for numNeighborsLoop = StartingNumNeighbors:EndingNumNeighbors
    NumberNeighbors = [NumberNeighbors numNeighborsLoop];
end
disp('Starting the search for Optimal Number of Neighbors');
ErrorsForValuesOfN = getAverageErrorForEachValueOfN(StartingNumNeighbors, EndingNumNeighbors, NumberFolds, y, X);

Dimensions_Errors = size(ErrorsForValuesOfN);
lowestError = 1000000;
lowestErrorNumNeighbors = -1;
    
for lowestErrorLoop = 1:Dimensions_Errors(2)
    if ErrorsForValuesOfN(1,lowestErrorLoop) < lowestError
        lowestError = ErrorsForValuesOfN(1,lowestErrorLoop);
        lowestErrorNumNeighbors = NumberNeighbors(1, lowestErrorLoop);
    end
end
OptimalNumNeighbors = lowestErrorNumNeighbors;
disp('Found the Optimal Number of Neighbors to be:');
disp(OptimalNumNeighbors);

%%% Use the Optimal K (K-NN) Value to Perform 10-Fold Cross Validation %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
NumberIterations = 10;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Starting 10 Iterations of 10-Fold Cross Validation Using Optimal Number of Neighbors');
iterationXfoldErrorMatrix = [];
for iterationLoop = 1:NumberIterations
    disp('Iteration Number:');
    disp(iterationLoop);
    CurIterationErrors = KFoldCrossValidation_Row(NumberFolds, OptimalNumNeighbors, y, X);
    iterationXfoldErrorMatrix = [iterationXfoldErrorMatrix; CurIterationErrors];
end

FinalAverageError = FindAverageErrorAcrossAllFolds(iterationXfoldErrorMatrix);
disp('Final Average Error:');
disp(FinalAverageError);

FinalStandardDeviation = FindErrorStandardDeviationAcrossAllFolds(iterationXfoldErrorMatrix, FinalAverageError);
disp('Standard Deviation:');
disp(FinalStandardDeviation);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Figure out a way to plot the errors for each value of num neighbors %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Need to plot the Average error for each value of K on scatter plot
scatter(NumberNeighbors,ErrorsForValuesOfN,[],[0,0,0],'filled');
ylim([0,0.35]);
xlabel('Number of Neighbors');
ylabel('Average Covid Severity Index Error');
title('Average Error in Predicted Covid Severity Index For 1 - 25 Neighbors');


%%% Clear Unwanted Variables %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear filename;
clear Dimensions_y;
clear Dimensions_X;
clear Num_Rows_X;
clear Num_Columns_X;
clear StartingNumNeighbors;
clear EndingNumNeighbors;
clear Dimensions_Errors;
clear numNeighborsLoop;
clear lowestError;
clear lowestErrorNumNeighbors;
clear lowestErrorLoop
clear NumberFolds;
clear NumberIterations;
clear CurIterationErrors;
clear iterationLoop;
% clear AverageError;
% clear finalAverageLoop;
% clear FinalAverageErrorSum;

%%% Extra Helper Functions %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function average = FindAverageErrorAcrossAllFolds(ErrorMatrix)
    DimensionsErrorMatrix = size(ErrorMatrix);
    IterationAverageErrors = [];
    
    % Each row represents an iteration
    % Each colums represents the error for a given fold in the iteration
    % Average all folds of a given iteration
    % Then average all iterations
    
    for averageErrorPerFoldLoop = 1:DimensionsErrorMatrix(1)
        curIterationErrorVector = ErrorMatrix(averageErrorPerFoldLoop,:);
        DimensionsCurIterationErrorVector = size(curIterationErrorVector);
        
        ErrorSum = 0;
        for innerLoop = 1:DimensionsCurIterationErrorVector(2)
            ErrorSum = ErrorSum + curIterationErrorVector(1,innerLoop);
        end
        
        AverageIterationError = ErrorSum / DimensionsCurIterationErrorVector(2);
        IterationAverageErrors = [IterationAverageErrors AverageIterationError];
    end
    
    % We now have a row vector containing the average error acrosds all folds
    % for each iteration
    DimensionsIterationAverageErrors = size(IterationAverageErrors);
    
    FinalAverageSum = 0;
    for finalAverageLoop = 1:DimensionsIterationAverageErrors(2)
        FinalAverageSum = FinalAverageSum + IterationAverageErrors(1,finalAverageLoop);
    end
    
    average = FinalAverageSum / DimensionsIterationAverageErrors(2);
end

function standardDeviation = FindErrorStandardDeviationAcrossAllFolds(ErrorMatrix, AverageError)
    DimensionsErrorMatrix = size(ErrorMatrix);
    NumberElements = DimensionsErrorMatrix(1) * DimensionsErrorMatrix(2);
    
    StandardDeviationComponentSum = 0;
    
    for rowsLoop = 1:DimensionsErrorMatrix(1)
        for columnsLoop = 1:DimensionsErrorMatrix(2)
            CurErrorValue = ErrorMatrix(rowsLoop,columnsLoop);
            CurDifferenceSquared = ((CurErrorValue - AverageError)^2);
            StandardDeviationComponentSum = StandardDeviationComponentSum + CurDifferenceSquared;
        end
    end
    
    SquaredQuotient = StandardDeviationComponentSum / (NumberElements - 1);

    standardDeviation = sqrt(SquaredQuotient);
end

function optimalKValue = findOptimalNumberNeighbors(StartingNumNeighbors, EndingNumNeighbors, NumFolds, ResponseVector, FeatureMatrix)
    Errors = [];
    NumNeighbors = [];
    for optimalNumNeighborsLoop = StartingNumNeighbors:EndingNumNeighbors
        curError = KFoldCrossValidation(NumFolds, optimalNumNeighborsLoop, ResponseVector, FeatureMatrix);
        Errors = [Errors curError];
        NumNeighbors = [NumNeighbors optimalNumNeighborsLoop];
    end
    
    Dimensions_Errors = size(Errors);
    lowestError = 1000000;
    lowestErrorNumNeighbors = -1;
    
    for lowestErrorLoop = 1:Dimensions_Errors(2)
        if Errors(1,lowestErrorLoop) < lowestError
            lowestError = Errors(1,lowestErrorLoop);
            lowestErrorNumNeighbors = NumNeighbors(1, lowestErrorLoop);
        end
    end
    optimalKValue = lowestErrorNumNeighbors;
end

function errorVector = getAverageErrorForEachValueOfN(StartingNumNeighbors, EndingNumNeighbors, NumFolds, ResponseVector, FeatureMatrix)
    Errors = [];
    NumNeighbors = [];
    for optimalNumNeighborsLoop = StartingNumNeighbors:EndingNumNeighbors
        curError = KFoldCrossValidation(NumFolds, optimalNumNeighborsLoop, ResponseVector, FeatureMatrix);
        Errors = [Errors curError];
        NumNeighbors = [NumNeighbors optimalNumNeighborsLoop];
    end
    errorVector = Errors;
end

function error = KFoldCrossValidation(K, N, ResponseVector, OriginalFeatureMatrix)
    % - We are going to split the samples into 10 even subsets
    KCV_Subsets = getKEvenSubsetsRandom(K, ResponseVector, OriginalFeatureMatrix);
    
    Errors = [];

    for testingLoop = 1:K

        CurTrainingSubsetX = [];
        CurTrainingSubsetY = [];

        CurTestingSubsetX = [];
        CurTestingSubsetY = [];

        for innerLoop = 1:K
            CurSubset = KCV_Subsets{1, innerLoop};
            CurSubsetDimensions = size(CurSubset);
            if innerLoop == testingLoop
                for innerInnerLoop = 1:CurSubsetDimensions(1)
                    curRow = CurSubset(innerInnerLoop,:);
                    CurX = curRow(1,1:(end-1));
                    CurY = curRow(1,end);
                    CurTestingSubsetX = [CurTestingSubsetX; CurX];
                    CurTestingSubsetY = [CurTestingSubsetY; CurY];
                end
            else
                for innerInnerLoop = 1:CurSubsetDimensions(1)
                    curRow = CurSubset(innerInnerLoop,:);
                    CurX = curRow(1,1:(end-1));
                    CurY = curRow(1,end);
                    CurTrainingSubsetX = [CurTrainingSubsetX; CurX];
                    CurTrainingSubsetY = [CurTrainingSubsetY; CurY];
                end
            end
        end
        
        DimensionsTraining = size(CurTrainingSubsetX);
        DimensionsTesting = size(CurTestingSubsetX);
        
        CurFoldPredictedResponses = predictSamples(N, CurTrainingSubsetX, CurTrainingSubsetY, CurTestingSubsetX);
        
        ErrorSum = 0;
       
        for accuracyLoop = 1:DimensionsTesting(1)
            ActualResponse = CurTestingSubsetY(accuracyLoop,1);
            PredictedResponse = CurFoldPredictedResponses(accuracyLoop,1);
            curError = abs(PredictedResponse - ActualResponse);
            ErrorSum = ErrorSum + curError;
        end
        
        AverageError = ErrorSum / DimensionsTesting(1);
        disp(AverageError);
        Errors = [Errors AverageError];
    end
    
    FinalErrorSum = 0;
    for finalErrorLoop = 1:K
        FinalErrorSum = FinalErrorSum + Errors(1,finalErrorLoop);
    end

    FinalAverageError = FinalErrorSum / K;
    error = FinalAverageError;
end

function errors = KFoldCrossValidation_Row(K, N, ResponseVector, OriginalFeatureMatrix)
    % - We are going to split the samples into 10 even subsets
    KCV_Subsets = getKEvenSubsetsRandom(K, ResponseVector, OriginalFeatureMatrix);
    
    Errors = [];

    for testingLoop = 1:K

        CurTrainingSubsetX = [];
        CurTrainingSubsetY = [];

        CurTestingSubsetX = [];
        CurTestingSubsetY = [];

        for innerLoop = 1:K
            CurSubset = KCV_Subsets{1, innerLoop};
            CurSubsetDimensions = size(CurSubset);
            if innerLoop == testingLoop
                for innerInnerLoop = 1:CurSubsetDimensions(1)
                    curRow = CurSubset(innerInnerLoop,:);
                    CurX = curRow(1,1:(end-1));
                    CurY = curRow(1,end);
                    CurTestingSubsetX = [CurTestingSubsetX; CurX];
                    CurTestingSubsetY = [CurTestingSubsetY; CurY];
                end
            else
                for innerInnerLoop = 1:CurSubsetDimensions(1)
                    curRow = CurSubset(innerInnerLoop,:);
                    CurX = curRow(1,1:(end-1));
                    CurY = curRow(1,end);
                    CurTrainingSubsetX = [CurTrainingSubsetX; CurX];
                    CurTrainingSubsetY = [CurTrainingSubsetY; CurY];
                end
            end
        end
        
        DimensionsTraining = size(CurTrainingSubsetX);
        DimensionsTesting = size(CurTestingSubsetX);
        
        CurFoldPredictedResponses = predictSamples(N, CurTrainingSubsetX, CurTrainingSubsetY, CurTestingSubsetX);
        
        ErrorSum = 0;
       
        for accuracyLoop = 1:DimensionsTesting(1)
            ActualResponse = CurTestingSubsetY(accuracyLoop,1);
            PredictedResponse = CurFoldPredictedResponses(accuracyLoop,1);
            curError = abs(PredictedResponse - ActualResponse);
            ErrorSum = ErrorSum + curError;
        end
        
        AverageError = ErrorSum / DimensionsTesting(1);
        disp(AverageError);
        Errors = [Errors AverageError];
    end

    errors = Errors;
end

function predictedResponses = predictSamples(K, TrainingFeatures, TrainingResponses, TestingFeatures)

    DimensionsTrainingFeatures = size(TrainingFeatures);
    DimensionsTrainingResponses = size(TrainingResponses);
    DimensionsTestingFeatures = size(TestingFeatures);

    MinMaxValuesOfFeatures = [];
    
    % Create an initial vector to hold all min and max values, and initialize
    % with extreme numbers to ensure they get overwritten
    for initialLoop = 1:DimensionsTrainingFeatures(2)
        MinMaxValuesOfFeatures = [MinMaxValuesOfFeatures 1000000 -1000000];
    end
    
    
    % Go through the entire feature matrix and find the minimum and maximum
    % values for each feature, update the MinMaxValuesOfFeatures vector
    for minMaxLoop = 1:DimensionsTrainingFeatures(1)
        curFeatureVector = TrainingFeatures(minMaxLoop, :);
        curFeatureVectorDimensions = size(curFeatureVector); % 1 x 27
        for innerLoop = 1:curFeatureVectorDimensions(2)
            curFeature = curFeatureVector(1,innerLoop);
            featureMinIndex = ((innerLoop - 1) * 2) + 1;
            featureMaxIndex = ((innerLoop - 1) * 2) + 2;
            if curFeature < MinMaxValuesOfFeatures(1,featureMinIndex)
                MinMaxValuesOfFeatures(1,featureMinIndex) = curFeature;
            end

            if curFeature > MinMaxValuesOfFeatures(1,featureMaxIndex)
                MinMaxValuesOfFeatures(1,featureMaxIndex) = curFeature;
            end
        end
    end
    
    % Normalize all the training samples with min-max normalization
    NormalizedTestingFeatureMatrix = [];
    % x_norm = x_new - x_min / x_max - m_min
    for testingFeatureLoop = 1:DimensionsTestingFeatures(1)
        NormalizedTestingFeatureVector = [];
        
        for xNewNormLoop = 1:DimensionsTrainingFeatures(2)
            curFeatureValue = TestingFeatures(testingFeatureLoop,xNewNormLoop);

            featureMinIndex = ((xNewNormLoop - 1) * 2) + 1;
            featureMaxIndex = ((xNewNormLoop - 1) * 2) + 2;
            featureMinValue = MinMaxValuesOfFeatures(1,featureMinIndex);
            featureMaxValue = MinMaxValuesOfFeatures(1,featureMaxIndex);

            NormalizedFeature = ((curFeatureValue - featureMinValue) / (featureMaxValue - featureMinValue));
            NormalizedTestingFeatureVector = [NormalizedTestingFeatureVector NormalizedFeature];
        end
        NormalizedTestingFeatureMatrix = [NormalizedTestingFeatureMatrix; NormalizedTestingFeatureVector];
    end
    
    % Now that we have the normalized vectors, go through all the Testing
    % samples and predict their responses by calculating a weighted average
    % of the K-Closest Training samples

    PredictedTestingSampleResponses = [];

    for testingSamplesPredictionLoop = 1:DimensionsTestingFeatures(1)
        curTestingVector = NormalizedTestingFeatureMatrix(testingSamplesPredictionLoop,:);
 
        KNearestSamples = {};
        % Go through each row of the training feature matrix
        for mainLoop = 1:DimensionsTrainingFeatures(1)
            curFeatureVector = TrainingFeatures(mainLoop, :);
            curFeatureVectorDimensions = size(curFeatureVector);

            % Normalize the current row using min-max normalization and the values we 
            % previously found
            normalizedCurFeatureVector = [];
            for innerMainLoop1 = 1:curFeatureVectorDimensions(2)
                curFeatureValue = curFeatureVector(1,innerMainLoop1);

                featureMinIndex = ((innerMainLoop1 - 1) * 2) + 1;
                featureMaxIndex = ((innerMainLoop1 - 1) * 2) + 2;
                featureMinValue = MinMaxValuesOfFeatures(1,featureMinIndex);
                featureMaxValue = MinMaxValuesOfFeatures(1,featureMaxIndex);

                NormalizedFeature = ((curFeatureValue - featureMinValue) / (featureMaxValue - featureMinValue));
                normalizedCurFeatureVector = [normalizedCurFeatureVector NormalizedFeature];
            end
        
            % Calculate the euclidean distance of the current row and x_new
            curEuclideanDistance = 0;
            for innerMainLoop2 = 1:curFeatureVectorDimensions(2)
                curFeatureValue = normalizedCurFeatureVector(1,innerMainLoop2);
                personalFeatureValue = curTestingVector(1,innerMainLoop2);
                curEuclideanDistanceComponent = ((curFeatureValue - personalFeatureValue)^(2));
                curEuclideanDistance = curEuclideanDistance + curEuclideanDistanceComponent;
            end
            curEuclideanDistance = sqrt(curEuclideanDistance);
       
            sizeKNearestSamples = size(KNearestSamples);
            if sizeKNearestSamples(2) < K
                curNode.euclideanDistance = curEuclideanDistance;
                curNode.response = TrainingResponses(mainLoop,1);
                curNode.index = mainLoop;
                curNode.originalFeatureVector = TrainingFeatures(mainLoop,:);

                KNearestSamples{end+1} = curNode;
                newSizeKNearestSamples = size(KNearestSamples);
                % Sort the collection when we reach K samples to begin the
                % comparison for remainder of the samples
                if newSizeKNearestSamples(2) == K
                    KNearestSamples = sortCollectionByEuclidenDistance(KNearestSamples);
                end
            else
                % Make the first index the closest, so the last index will always
                % be used as the basis for comparison
                furthestOfKClosest = KNearestSamples{end};
                furthestOfKClosestDistance = furthestOfKClosest.euclideanDistance;
                
                if curEuclideanDistance < furthestOfKClosestDistance
                    curNode.euclideanDistance = curEuclideanDistance;
                    curNode.response = TrainingResponses(mainLoop,1);
                    curNode.index = mainLoop;
                    curNode.originalFeatureVector = TrainingFeatures(mainLoop,:);

                    newTempKNearestSamples = {};
                    Dimensions_KNearestSamples = size(KNearestSamples);
                    for deleteLoop = 1:(Dimensions_KNearestSamples(2) - 1)
                        newTempKNearestSamples{end+1} = KNearestSamples{deleteLoop};
                    end

                    newTempKNearestSamples{end+1} = curNode;
                    
                    % There is no need to sort here
                    % We know the new value is at the last index
                    %Simply go throough all the values and find the index where
                    %the new value is supposed to go.
                    finalTempKNearestSamples = {};
                    FoundIndex = Dimensions_KNearestSamples(2);
                    NewFoundIndex = false;
                    for searchLoop = 1:(Dimensions_KNearestSamples(2) - 1)
                        CurElement = KNearestSamples{searchLoop};
                        CurNode = CurElement(1);
                        CurDistance = CurNode.euclideanDistance;
                        if curEuclideanDistance < CurDistance
                            if (NewFoundIndex == false)
                                FoundIndex = searchLoop;
                                NewFoundIndex = true;
                            end
                        end
                    end
                
                    for placementLoop = 1:FoundIndex-1
                        finalTempKNearestSamples{end+1} = newTempKNearestSamples{placementLoop};
                    end
                
                    finalTempKNearestSamples{end+1} = curNode;
                
                    for placementLoop2 = FoundIndex:(Dimensions_KNearestSamples(2) - 1)
                        finalTempKNearestSamples{end+1} = newTempKNearestSamples{placementLoop2};
                    end
                    KNearestSamples = finalTempKNearestSamples;
                else
                    % Do nothing, too far away to be considered
                end
            end
        end
        
        % Where are we?
        % We now have a collection of the K nearest training samples to
        % the current testing sample.
        
        % We now need to predict the response of this current testing
        % vector via a weighted average where the weights are given by 
        % w(k) = 1 / euclideanDistance(xtest , x_k)
        
        WeightedAverageSum = 0;
        SumOfWeights = 0;
        for predictionLoop = 1:K
            curNearestNeighbor = KNearestSamples{predictionLoop};
            %     disp(curNearestNeighbor);
            curDistance = curNearestNeighbor.euclideanDistance;
            curResponse = curNearestNeighbor.response;
            curWeight = exp(-1 * curDistance);
            
            curWeightedAverageSumComponent = curResponse * curWeight;
            WeightedAverageSum = WeightedAverageSum + curWeightedAverageSumComponent;
            SumOfWeights = SumOfWeights + curWeight;
        end
        
        FinalCurrentTestingVectorPrediction = WeightedAverageSum / SumOfWeights;
        
        PredictedTestingSampleResponses = [PredictedTestingSampleResponses; FinalCurrentTestingVectorPrediction];
    end
    predictedResponses = PredictedTestingSampleResponses;
end

function subsets = getKEvenSubsetsRandom(K, ResponseVector, OriginalFeatureMatrix)
    
    NumTotalSamples = size(ResponseVector);
    intK = int16(K);
    SubsetQuantity = idivide(NumTotalSamples(1), intK,'floor');
    LeftoverStart = mod(NumTotalSamples(1), intK);
    LeftoverAdded = 0;
    
    K_Subsets = {};
    
    RandomIndexPermutation = randperm(NumTotalSamples(1));

    for numSubsetsLoop = 1:K
        
        curSubset = [];
        
        startIndex = (((numSubsetsLoop - 1) * SubsetQuantity) + LeftoverAdded) + 1;
        stopIndex = startIndex + SubsetQuantity - 1;
        
        if (LeftoverStart - LeftoverAdded) > 0
            stopIndex = stopIndex + 1;
            LeftoverAdded = LeftoverAdded + 1;
        end
        
        for innerLoop = startIndex:stopIndex
            curRandomIndex = RandomIndexPermutation(1,innerLoop);
            curXVector = OriginalFeatureMatrix(curRandomIndex,:);
            Cur_y_value = ResponseVector(curRandomIndex,1);
            curSubset = [curSubset; curXVector Cur_y_value];
        end
        
        K_Subsets{end + 1} = curSubset;
        
    end

    subsets = K_Subsets;
end

% Description: Uses Merge Sort to sort the unsortedCollection into a sorted
% collection in ascending order. This is used in the K-Nearest Neighbors
% implementation to constantly ensure we can identify the nearest K nodes
% at any time
% 
% Parameters:
% - unsortedCollection - The unsorted collection to sort and return in ascending order 
%
function sortedCollection = sortCollectionByEuclidenDistance(unsortedCollection)
    Distances = [];
    DistanceIndices = [];
    unsortedCollectionDimension = size(unsortedCollection);
    
    for distanceLoop = 1:unsortedCollectionDimension(2)
        curNode = unsortedCollection{distanceLoop};
        convertedCurNode = curNode(1);
        
        Distances = [Distances convertedCurNode.euclideanDistance];
        DistanceIndices = [DistanceIndices distanceLoop];
    end
    
    % Distances now is a vector of all the euclidean distances, perform an
    % ascending sort algorithm while tracking indices
    
    % Selection sort
    for selectionSortLoop = 1:(unsortedCollectionDimension(2) - 1)
        minValue = Distances(1,selectionSortLoop);
        minIndex = selectionSortLoop;
        for innerLoop = (selectionSortLoop + 1):unsortedCollectionDimension(2)
            if Distances(1,innerLoop) < minValue
                minValue = Distances(1,innerLoop);
                minIndex = innerLoop;
            end
        end
        
        curValue = Distances(1,selectionSortLoop);
        curIndex = DistanceIndices(1,selectionSortLoop);
        
        Distances(1,selectionSortLoop) = minValue;
        DistanceIndices(1,selectionSortLoop) = DistanceIndices(1,minIndex);
        
        Distances(1,minIndex) = curValue;
        DistanceIndices(1,minIndex) = curIndex;
    end
    
    % At the end of this we will have an array with the ordering of the
    % smallest to largest euclidean distances [5 3 2 4 1]. Need to go into
    % the collection and select the index and place it in the correct spot
    % in new collection
    
    newCollection = {};
    for finalSortLoop = 1:unsortedCollectionDimension(2)
        curIndex = DistanceIndices(1,finalSortLoop);
        newCollection{end+1} = unsortedCollection{curIndex};
    end
    sortedCollection = newCollection;
end

function radixSortedCollection = radixSortWithNodes(unsortedCollection)
    NumberDecimals = 10;
    Dimensions_Unsorted = size(unsortedCollection);

    MaximumMagnitude = 0;
    NormalizedUnsortedCollection = {};
    
    % Go through the collection, normalize each value to a consistent
    % number of decminal places, and understand the maximum magnitude
    for NormalizingLoop = 1:Dimensions_Unsorted(2)
        CurElement = unsortedCollection{NormalizingLoop};
        CurObject = CurElement(1);
        CurValue = CurObject.euclideanDistance;
        
        NormalizedCurValue = round(CurValue, NumberDecimals);
        NormalizedUnsortedCollection{end+1} = CurElement;
        
        flag = true;
        startMagnitude = 1;
        TempMaxMagnitude = 0;
        while flag == true
            % 5 / 1 = 5 > 0 --> Add 1
            % 5 / 10 = 0 !> 0 --> Stop for this number
            % 50 /1 = 50 > 0 --> Add 1
            % 50 / 10 = 5 > 0 --> Add 1
            % 50 / 100 = 0 !> 0 --> Stop for this number
            CurQuotient = floor(CurValue / startMagnitude);
            if CurQuotient > 0
                TempMaxMagnitude = TempMaxMagnitude + 1;
                startMagnitude = startMagnitude * 10;
            else
                if TempMaxMagnitude > MaximumMagnitude
                    MaximumMagnitude = TempMaxMagnitude;
                end
                flag = false;
            end
        end
    end
    
    % At the end of this, all values will contain 10 decimal places, as
    % well as a maximum of MaximumMagnitude places in front of the decimal
    
    IntermediateCollection = unsortedCollection;
    % Go through the collection and sort the numbers by their decimal
    for DecimalLoop = 1:(NumberDecimals + MaximumMagnitude)
        CurrentDivisor = 10^(-1 * (NumberDecimals - DecimalLoop + 1));
        CurIndexCounts = [0 0 0 0 0 0 0 0 0 0];
        for innerDecimalLoop = 1:Dimensions_Unsorted(2)
            
            CurElement = IntermediateCollection{innerDecimalLoop};
            CurObject = CurElement(1);
            CurValue = CurObject.euclideanDistance;

            CurIntermediateValue = CurValue / CurrentDivisor;
            CurFinalValue = floor(mod(CurIntermediateValue, 10));
            CurFinalModValue = idivide(CurFinalValue, int16(1)) + 1;
            CurIndexCounts(1,CurFinalModValue) = CurIndexCounts(1,CurFinalModValue) + 1;
        end
        
        % We now have a table with # occurences, determine the index values
        IndexTable = [1 0 0 0 0 0 0 0 0 0];
        for IndexTableLoop = 2:10
            IndexTable(1,IndexTableLoop) = IndexTable(1,IndexTableLoop-1) + CurIndexCounts(1,IndexTableLoop-1);
        end
        
        % We now have a table with the starting indices for each value at
        % the current place
        TempIntermediateCollection = IntermediateCollection;
        for PlacementLoop = 1:Dimensions_Unsorted(2)
            
            CurElement = IntermediateCollection{PlacementLoop};
            CurObject = CurElement(1);
            CurValue = CurObject.euclideanDistance;
            
            CurIntermediateValue = CurValue / CurrentDivisor;
            CurFinalValue = floor(mod(CurIntermediateValue, 10)) + 1;
            CurIndex = IndexTable(1,CurFinalValue);
            TempIntermediateCollection{1,CurIndex} = CurElement;
            IndexTable(1,CurFinalValue) = IndexTable(1,CurFinalValue) + 1;
        end
        IntermediateCollection = TempIntermediateCollection;
    end
    
    radixSortedCollection = IntermediateCollection;
end