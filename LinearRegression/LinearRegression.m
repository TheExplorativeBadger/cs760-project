%%% Problem Setup %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
filename = 'FinalCombinedFeaturesWithSeverity.csv';
OriginalFile = dlmread(filename, ';');

% y = covid severity index
y = OriginalFile(:,1);
Dimensions_y = size(y);
% X = 1-Bias|Demographic-Feature-Matrix
X = [ones(Dimensions_y(1),1) OriginalFile(:,2:end)];
Dimensions_X = size(X);
Num_Rows_X = Dimensions_X(1);
Num_Columns_X = Dimensions_X(2);

%%% Find MLE Theta  %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is the MLE(Theta) of Homoskedastic
Theta_Hat = inv(X' * X) * X' * y;

% This is the MLE(Sigma) of Homoskedastic
sigma_squared = 1/Num_Rows_X * (y - X * Theta_Hat)' * (y - X * Theta_Hat);

% This is the MLE(Sigma) of Heteroskedastic
Sigma_Hat = (y - X * Theta_Hat) * (y - X * Theta_Hat)';

%%% Iterative Heteroskedastic MLE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NumberIterations = 1000;
% % Iterative Heteroskedastic Approach
% Theta_Hat_Iterative = zeros(Dimensions_X(2), 1);
% Sigma_Hat_Iterative = eye(Dimensions_y(1));
% for loops = 1:NumberIterations
%     Theta_Hat_Iterative_New = inv(X' * inv(Sigma_Hat_Iterative) * X) * X' * inv(Sigma_Hat_Iterative) * y;
%     Sigma_Hat_Iterative = (y - (X * Theta_Hat_Iterative)) * (y - (X * Theta_Hat_Iterative))';
%     Theta_Hat_Iterative = Theta_Hat_Iterative_New;
% end

%%% Build a new Feature Vector %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% New Sample Details
% x_New = [1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1;];
% num_samples = 1;

% Set the threshold to 0.05, find the z_score
alpha = 0.05;
% z_score = abs(norminv(alpha/2));

%%% Predict the Responses %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Find the new estimated response MLE
% y_Hat = x_New' * Theta_Hat;
% y_Hat_Hetero = x_New' * Theta_Hat_Iterative;

%%% Homoskedastic Confidence Interval %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% variance_homo = sigma_squared * x_New' * inv(X' * X) * x_New;
% std_dev_homo = sqrt(variance_homo);
% 
% tau_Homo = z_score * (std_dev_homo / sqrt(num_samples));
% 
% confidence_upper_homo = y_Hat + z_score * (std_dev_homo / sqrt(num_samples));
% confidence_lower_homo = y_Hat - z_score * (std_dev_homo / sqrt(num_samples));
% Confidence_Interval_Homoskedastic = [confidence_lower_homo, confidence_upper_homo];

%%% Heteroskedastic Confidence Interval %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% variance_hetero = x_New' * inv(X' * inv(Sigma_Hat_Iterative) * X) * x_New;
% std_dev_hetero = sqrt(variance_hetero);
%
% tau_Hetero = z_score * (std_dev_hetero / sqrt(num_samples));
%
% confidence_upper_hetero = y_Hat + z_score * (std_dev_hetero / sqrt(num_samples));
% confidence_lower_hetero = y_Hat - z_score * (std_dev_hetero / sqrt(num_samples));
% Confidence_Interval_Heteroskedastic = [confidence_lower_hetero, confidence_upper_hetero];

%%% Determine Significant Features %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cov_matrix = sigma_squared * inv(X' * X);

SignificanceLevels_Homoskedastic = [];
SignificantFeatureIndices_Homoskedastic = [];

% Inverse tail of chi squared (alpha = 0.05) = 3.838
SignificanceThreshold = 3.838;

for significanceLoop = 1:Num_Columns_X
    curSignificanceLevel = Theta_Hat(significanceLoop)^2 / cov_matrix(significanceLoop, significanceLoop);
    SignificanceLevels_Homoskedastic = [SignificanceLevels_Homoskedastic curSignificanceLevel];

    if curSignificanceLevel > SignificanceThreshold
        SignificantFeatureIndices_Homoskedastic = [SignificantFeatureIndices_Homoskedastic significanceLoop];
    end
end

%%% Run 10 sets of 10-Fold Cross Validation to Determine Error %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numIterations = 10;

AverageErrors = [];
for iterationLoop = 1:numIterations
    AverageError = KFoldCrossValidation(10, y, X);
    AverageErrors = [AverageErrors AverageError];
end

FinalAverageErrorSum = 0;
for finalAverageLoop = 1:numIterations
    FinalAverageErrorSum = FinalAverageErrorSum + AverageErrors(1,finalAverageLoop);
end

FinalAverageError = FinalAverageErrorSum / numIterations;

%%% Clear Unwanted Variables %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear filename;
clear Dimensions_y;
clear Dimensions_X;
clear Num_Rows_X;
clear Num_Columns_X;
clear significanceLoop;
clear curSignificanceLevel;
clear iterationLoop;
clear finalAverageLoop;
clear FinalAverageErrorSum;

%%% Extra Helper Functions %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function error = KFoldCrossValidation(K, ResponseVector, OriginalFeatureMatrix)
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

        % We now have a clearly defined matrix of testing and training
        % samples, and their corresponding Responses

        DimensionsTraining = size(CurTrainingSubsetX);
        DimensionsTesting = size(CurTestingSubsetX);

        Theta_Hat = inv(CurTrainingSubsetX' * CurTrainingSubsetX) * CurTrainingSubsetX' * CurTrainingSubsetY;

        ErrorSum = 0;

        for accuracyLoop = 1:DimensionsTesting
            curTestingSample = CurTestingSubsetX(accuracyLoop,:);
            curTestingResponse = CurTestingSubsetY(accuracyLoop,1);

            curTestingSampleResponsePrediction = curTestingSample * Theta_Hat;

            curError = abs(curTestingSampleResponsePrediction - curTestingResponse);
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

function subsets = getKEvenSubsetsSequential(K, ResponseVector, OriginalFeatureMatrix)

    NumTotalSamples = size(ResponseVector);
    intK = int16(K);

    SubsetQuantity = idivide(NumTotalSamples(1), intK,'floor');
    LeftoverStart = mod(NumTotalSamples(1), intK);
    LeftoverAdded = 0;

    K_Subsets = {};

    Dimensions_X = size(OriginalFeatureMatrix);
    Num_Rows_X = Dimensions_X(1);

    for kSubsetsLoop = 1:K
        CurSubset = [];

        startIndex = (((kSubsetsLoop - 1) * SubsetQuantity) + LeftoverAdded) + 1;
        stopIndex = startIndex + SubsetQuantity - 1;


        if (LeftoverStart - LeftoverAdded) > 0
            stopIndex = stopIndex + 1;
            LeftoverAdded = LeftoverAdded + 1;
        end

        for innerLoop = startIndex:stopIndex
            curXVector = OriginalFeatureMatrix(innerLoop,:);
            Cur_y_value = ResponseVector(innerLoop);
            CurSubset = [CurSubset; curXVector Cur_y_value];
        end

        K_Subsets{end + 1} = CurSubset;
    end

    subsets = K_Subsets;
end