function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
steps=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
lowest_value = 0;
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
m=size(steps,2);
for s=1:m
    for c=1:m
        c_val=steps(c);
        s_val=steps(s);
        model= svmTrain(X, y, c_val, @(x1, x2) gaussianKernel(x1, x2, s_val));
        predictions = svmPredict(model, Xval);
        temp = mean(double(predictions ~= yval));
        if (c==1 && s==1)
            lowest_value = temp;
        end
        if (temp < lowest_value)
            lowest_value = temp;
            C=c_val;
            sigma=s_val;
        end
    end
end






% =========================================================================

end
