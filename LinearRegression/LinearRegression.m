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

%%% Find Theta Hat %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Theta_Hat = (((X' * X)^(-1)) * X') * y;