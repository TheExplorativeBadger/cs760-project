%%% Implementation Configurations %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Distance-weighted Nearest Neighbors with
% All values min-max normalized
% Distance measure: Euclidean
% w(k) = 1 / distance(x_new , x_k)

%%% TODO %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% - Modify this to work with regression 
% - Change the K-Subsets to just 10 even subsets
% - Implement the accuracy determination using the regression version of
% accuracy

%%% Problem Setup %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% filename = 'FinalCombinedFeaturesWithSeverity.csv';
filename = 'titanic_data_numbers.csv';
OriginalFile = csvread(filename);

% y = covid severity index 
y = OriginalFile(:,1);
disp(y);
Dimensions_y = size(y);
% X = 1-Bias|Demographic-Feature-Matrix
X = [OriginalFile(:,2:end)];
Dimensions_X = size(X);
Num_Rows_X = Dimensions_X(1);
Num_Columns_X = Dimensions_X(2);




%%% Perform K Fold Cross Validation to Determine Best K %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

K = KFoldCrossValidation(10, y, X);

%%% Create your own personal feature vector to test %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

PersonalFeatureVector = [1 0 25 1 0 27.50];

%%% Classify the personal feature vector using optimal K %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Classification = classifyNewSample(K, X, y, PersonalFeatureVector);

%%% Plot of K values, from before KFCV implemented %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% scatter(X_Coordinates,OverallResponses);
% ylim([-0.2,1.2]);

%%% Useful Functions %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


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

% Description: Returns a classification pediction for the TestingVector
% based on a K-Nearest Neighbors implementation that uses the
% TrainingFeatures and TrainingResponses to find the K closest samples and
% determine the best prediction
% 
% Parameters:
% - K - The K value in K-Nearest Neighbors to use for this classification
% - TrainingFeatures - The features to use as training samples when
% calculating the euclidean distance
% - TrainingResponses - The corresponding response for each training
% feature vector
% - TestingVector - The feature vector you wish to predict
%
function classification = classifyNewSample(K, TrainingFeatures, TrainingResponses, TestingVector)

    DimensionsTrainingFeatures = size(TrainingFeatures);
    DimensionsTrainingResponses = size(TrainingResponses);

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
        curFeatureVectorDimensions = size(curFeatureVector); % 1 x 6
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
    
    NormalizedTestingFeatureVector = [];
    % x_norm = x_new - x_min / x_max - m_min
    for xNewNormLoop = 1:DimensionsTrainingFeatures(2)
        curFeatureValue = TestingVector(1,xNewNormLoop);

        featureMinIndex = ((xNewNormLoop - 1) * 2) + 1;
        featureMaxIndex = ((xNewNormLoop - 1) * 2) + 2;
        featureMinValue = MinMaxValuesOfFeatures(1,featureMinIndex);
        featureMaxValue = MinMaxValuesOfFeatures(1,featureMaxIndex);

        NormalizedFeature = ((curFeatureValue - featureMinValue) / (featureMaxValue - featureMinValue));
        NormalizedTestingFeatureVector = [NormalizedTestingFeatureVector NormalizedFeature];
    end
    
 
    % Go through the response vector and get all the unique classifications
    % Keep track of number of unique and unique values.
    UniqueClassificationCounter = 0;
    UniqueClassifications = [];
    for classificationLoop = 1:DimensionsTrainingResponses(1)
        curResponse = TrainingResponses(classificationLoop,1);

        alreadyExists = false;
        UniqueClassificationsDimensions = size(UniqueClassifications);

        for innerClassificationLoop = 1:UniqueClassificationsDimensions(2)
            if UniqueClassifications(1,innerClassificationLoop) == curResponse
                alreadyExists = true;
            end
        end

        if alreadyExists == false
            UniqueClassifications = [UniqueClassifications curResponse];
            UniqueClassificationCounter = UniqueClassificationCounter + 1;
        end
    end

    
    KNearestSamples = {};
    % Go through each row
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
            personalFeatureValue = NormalizedTestingFeatureVector(1,innerMainLoop2);
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
                % Do nothing, the current row is too far away
            end
        end
    end

    % By the end, we should have a collection with the K closest nodes, that each 
    % contain a euclidean distance and a label

    ClassificationCounts = [];
    for i = 1:UniqueClassificationCounter
        ClassificationCounts = [ClassificationCounts 0];
    end

    % Sum all the weights for each class, and the class with highest sum is the
    % prediction
    for predictionLoop = 1:K
        curNearestNeighbor = KNearestSamples{predictionLoop};
    %     disp(curNearestNeighbor);
        curDistance = curNearestNeighbor.euclideanDistance;
        curClassification = curNearestNeighbor.response;
        curWeight = exp(-1 * curDistance);

        for i = 1:UniqueClassificationCounter
            if curClassification == UniqueClassifications(1,i)
                ClassificationCounts(1,i) = ClassificationCounts(1,i) + curWeight;
            end
        end
    end

    HighestWeight = 0;
    HighestIndex = 0;
    for highestClassificationLoop = 1:UniqueClassificationCounter
        if ClassificationCounts(1,highestClassificationLoop) > HighestWeight
            HighestWeight = ClassificationCounts(1,highestClassificationLoop);
            HighestIndex = highestClassificationLoop;
        end
    end

    FinalClassification = UniqueClassifications(1,HighestIndex);

    classification = FinalClassification;
end

% Description: Splits the samples into K even subsets with the same number
% of samples with each class in each subset
% 
% Parameters:
% - K - The number of subsets to split the data into
% - ResponseVector - The vector of response values for each matching sample
% - OriginalFeatureMatrix - The collection of feature vectors to split into
% K subsets
%
function subsets = getKEvenSubsets(K, ResponseVector, OriginalFeatureMatrix)

    Dimensions_y = size(ResponseVector);
    
    intK = int16(K);

    yCounts = [0 0];
    for yLoop = 1:Dimensions_y(1)
        if ResponseVector(yLoop,1) == 0
            yCounts(1,1) = yCounts(1,1) + 1;
        else
            yCounts(1,2) = yCounts(1,2) + 1;
        end
    end
    
    SubsetQuantity_Y0 = idivide(yCounts(1,1),intK,'floor');
    Leftover_Y0_Start = mod(yCounts(1,1),intK);
    Leftover_Y0_Added = 0;
    SubsetQuantity_Y1 = idivide(yCounts(1,2),intK,'floor');
    Leftover_Y1_Start = mod(yCounts(1,2),intK);
    Leftover_Y1_Added = 0;

    % Go through the entire set of samples, X, and pick out the first SQ_Y#
    % samples from each class and create a subset - store these subsets
    % together - you will be isolating each so need easy access
    K_Subsets = {};

    Y0_Samples = [];
    Y1_Samples = [];
    
    Dimensions_X = size(OriginalFeatureMatrix);
    Num_Rows_X = Dimensions_X(1);
    
    for separationLoop = 1:Num_Rows_X
        Cur_x_vector = OriginalFeatureMatrix(separationLoop,:);
        Cur_y_value = ResponseVector(separationLoop,1);

        if Cur_y_value == 0
           Y0_Samples = [Y0_Samples; Cur_x_vector];
        else
           Y1_Samples = [Y1_Samples; Cur_x_vector];
        end
    end
    
    for kSubsetsLoop = 1:K
        CurSubset = [];

        startIndex_Y0 = (((kSubsetsLoop - 1) * SubsetQuantity_Y0) + Leftover_Y0_Added) + 1;
        stopIndex_Y0 = startIndex_Y0 + SubsetQuantity_Y0 - 1;

        if (Leftover_Y0_Start - Leftover_Y0_Added) > 0
            stopIndex_Y0 = stopIndex_Y0 + 1;
            Leftover_Y0_Added = Leftover_Y0_Added + 1;
        end
        
        for innerY0Loop = startIndex_Y0:stopIndex_Y0
            curXVector = Y0_Samples(innerY0Loop,:);
            Cur_y_value = 0;
            CurSubset = [CurSubset; curXVector Cur_y_value];
        end

        startIndex_Y1 = (((kSubsetsLoop - 1) * SubsetQuantity_Y1) + Leftover_Y1_Added) + 1;
        stopIndex_Y1 = startIndex_Y1 + SubsetQuantity_Y1 - 1;

        if (Leftover_Y1_Start - Leftover_Y1_Added) > 0
            stopIndex_Y1 = stopIndex_Y1 + 1;
            Leftover_Y1_Added = Leftover_Y1_Added + 1;
        end

        for innerY1Loop = startIndex_Y1:stopIndex_Y1
            curXVector = Y1_Samples(innerY1Loop,:);
            Cur_y_value = 1;
            CurSubset = [CurSubset; curXVector Cur_y_value];
        end

        K_Subsets{end + 1} = CurSubset;
    end
    
    subsets = K_Subsets;
end

% Description: Performs K-Fold Cross Validation on a K-Nearest Neighbors
% implementation to determine the optimal value of K to use in classifying
% new samples for K-Nearest Neighbors implementation
% 
% Parameters:
% - K - The number of subsets to split the data into
% - ResponseVector - The vector of response values for each matching sample
% - OriginalFeatureMatrix - The collection of feature vectors to split into
% K subsets
%
function optimalKValue = KFoldCrossValidation(K, ResponseVector, OriginalFeatureMatrix)
    % - We are going to split the samples into 10 even subsets
    KCV_Subsets = getKEvenSubsets(K, ResponseVector, OriginalFeatureMatrix);
    
    ChosenNeighborValues = [];
    ChosenNeighborValueAccuracy = [];
    
    OverallAccuracy = [];
    
    for kNNLoop = 1:50
        Accuracy = [];
        disp("Loop:");
        disp(kNNLoop);
        for accuracyLoop = 1:K
            disp("Subset:");
            disp(accuracyLoop);

            CurTrainingSubsetX = [];
            CurTrainingSubsetY = [];

            CurTestingSubsetX = [];
            CurTestingSubsetY = [];

            for innerLoop = 1:K
                CurSubset = KCV_Subsets{1, innerLoop};
                CurSubsetDimensions = size(CurSubset);
                if innerLoop == accuracyLoop
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

            % We now have a clearly defined matrix of testing and training
            % samples, and their corresponding Responses

            DimensionsTraining = size(CurTrainingSubsetX);
            DimensionsTesting = size(CurTestingSubsetX);
            
            CorrectPredictionCounter = 0;
            for innerKNNLoop = 1:DimensionsTesting(1)
                curTestingFeature = CurTestingSubsetX(innerKNNLoop,:);
                curClassification = classifyNewSample(kNNLoop, CurTrainingSubsetX, CurTrainingSubsetY, curTestingFeature);

                if curClassification == CurTestingSubsetY(innerKNNLoop,1)
                    CorrectPredictionCounter = CorrectPredictionCounter + 1;
                end
            end

            CurrentKAccuracy = CorrectPredictionCounter / DimensionsTesting(1);
            % Record the accuracy in Accuracy[]
            Accuracy = [Accuracy CurrentKAccuracy];
        end
        
        % We now have the accuracies for this value of K for all subsets
        % Average them
        DimensionsAccuracy = size(Accuracy);
        AccuracySum = 0;
        for averagingLoop = 1: DimensionsAccuracy(2)
            AccuracySum = AccuracySum + Accuracy(1,averagingLoop);
        end
        
        AverageAccuracy = AccuracySum / DimensionsAccuracy(2);
        OverallAccuracy = [OverallAccuracy AverageAccuracy];
    end
    
    DimensionsOverallAccuracy = size(OverallAccuracy);
    
    HighestOverallAccuracy = 0;
    HighestOverallAccuracyIndex = 0;
    
    for finalLoop = 1:DimensionsOverallAccuracy(2)
        if OverallAccuracy(1,finalLoop) > HighestOverallAccuracy
            HighestOverallAccuracy = OverallAccuracy(1,finalLoop);
            HighestOverallAccuracyIndex = finalLoop;
        end
    end
    
    
    disp(HighestOverallAccuracy);
    disp(HighestOverallAccuracyIndex);
    
    optimalKValue = HighestOverallAccuracyIndex;
end
