function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

J = sum(-1 * y .* log(sigmoid(X * theta)) + (y - 1) .* log(1 - sigmoid(X * theta))) / m + lambda * sum((theta .^ 2)(2:end)) / (2 * m);

%grad = X' * (sigmoid(X * theta) - y) / m + lambda * theta / m;
%grad(1) = grad(1) - lambda * theta(1) / m;
grad = X' * (sigmoid(X * theta) - y) / m;
grad(2:end) += (lambda * theta / m)(2:end);


% =============================================================

end
