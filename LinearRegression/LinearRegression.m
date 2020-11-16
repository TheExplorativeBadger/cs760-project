%%% Problem Setup %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
filename = 'FinalCombinedFeaturesWithSeverity.csv';
OriginalFile = csvread(filename);

% y = covid severity index 
y = OriginalFile(:,1);
Dimensions_y = size(y);
% X = 1-Bias|Demographic-Feature-Matrix
X = [ones(Dimensions_y(1),1) OriginalFile(:,2-end)];
Dimensions_X = size(X);
Num_Rows_X = Dimensions_X(1);
Num_Columns_X = Dimensions_X(2);

%%% Find MLE Theta  %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is the MLE(Theta) of Homoskedastic
Theta_Hat = inv((X' * X)) * X' * y;

% This is the MLE(Sigma) of Heteroskedastic
Sigma_Hat = (y - X * Theta_Hat) * (y - X * Theta_Hat)';

%%% Iterative Heteroskedastic MLE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
NumberIterations = 1000;

% Iterative Heteroskedastic Approach
Theta_Hat_Iterative = zeros(X_size(2), 1);
Sigma_Hat_Iterative = eye(y_size(1));
for loops = 1:NumberIterations
    Theta_Hat_Iterative_New = inv(X' * inv(Sigma_Hat_Iterative) * X) * X' * inv(Sigma_Hat_Iterative) * y;
    Sigma_Hat_Iterative = (y - (X * Theta_Hat_Iterative)) * (y - (X * Theta_Hat_Iterative))';
    Theta_Hat_Iterative = Theta_Hat_Iterative_New; 
end

%%% Build a new Feature Vector %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% New Sample Details
x_New = [1; 1; 1; 1];
num_samples = 1;

% Set the threshold to 0.05, find the z_score
alpha = 0.05;
z_score = abs(norminv(alpha/2));

%%% Predict the Responses %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Find the new estimated response MLE
y_Hat = x_New' * Theta_Hat;
y_Hat_Hetero = x_New' * Theta_Hat_Iterative;

%%% Homoskedastic Confidence Interval %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sigma_squared = 1/Num_Rows_X * (y - X * Theta_Hat)' * (y - X * Theta_Hat);
variance_homo = sigma_squared * x_New' * inv(X' * X) * x_New;
std_dev_homo = sqrt(variance_homo);

tau_Homo = z_score * (std_dev_homo / sqrt(num_samples));

confidence_upper_homo = y_hat + z_score * (std_dev_homo / sqrt(num_samples));
confidence_lower_homo = y_hat - z_score * (std_dev_homo / sqrt(num_samples));
Confidence_Interval_Homoskedastic = [confidence_lower_homo, confidence_upper_homo];

%%% Heteroskedastic Confidence Interval %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
variance_hetero = x_New' * inv(X' * inv(Sigma_Hat_Iterative) * X) * x_New;
std_dev_hetero = sqrt(variance_hetero);

tau_Hetero = z_score * (std_dev_hetero / sqrt(num_samples));

confidence_upper_hetero = y_Hat + z_score * (std_dev_hetero / sqrt(num_samples));
confidence_lower_hetero = y_Hat - z_score * (std_dev_hetero / sqrt(num_samples));
Confidence_Interval_Heteroskedastic = [confidence_lower_hetero, confidence_upper_hetero];

%%% Determine Significant Features %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cov_matrix = sigma_squared * inv(X' * X);

SignificanceLevels_Homoskedastic = [];
SignificantFeatureIndices_Homoskedastic = [];

SignificanceLevels_Heteroskedastic = [];
SignificantFeatureIndices_Heteroskedastic = [];

for significanceLoop = 1:Num_Columns_X
    curSignificanceLevel = Theta_Hat(significanceLoop) / cov_matrix(significanceLoop, significanceLoop);
    SignificanceLevels_Homoskedastic = [SignificanceLevels_Homoskedastic curSignificanceLevel];
    
    if curSignificanceLevel > tau
        SignificantFeatureIndices_Homoskedastic = [SignificantFeatureIndices_Homoskedastic significanceLoop];
    end
end
