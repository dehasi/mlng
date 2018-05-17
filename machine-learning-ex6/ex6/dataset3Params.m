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
x1 = [1 2 1]; x2 = [0 4 -1];
ex = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
pser = 0;
C_t = 0; 
sigma_t = 0;
n = length(ex);
for c = 1:n
    for s = 1:n
        C = ex(c);
        sigma = ex(s);
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        pred = svmPredict(model, Xval);
        
        er = eq(pred,yval);
        ser = sum(er);
        if (ser > pser) 
            C_t = C; sigma_t = sigma; pser = ser;
        end;
           
    end
end
disp([C_t, sigma_t, pser]);

C = C_t;
sigma = sigma_t;

% =========================================================================

end
