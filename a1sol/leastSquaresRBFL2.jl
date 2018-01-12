include("misc.jl")

function rbfBasis(X1, X2, sigma)
  return exp.(-distancesSquared(X1,X2)/(2*sigma^2))
end

function leastSquaresRBFL2(X, y, l, sigma)
	# Make RBF basis
	n = size(X,1)
	Z = rbfBasis(X, X, sigma)

	# Find regression weights minimizing squared error with L2 regularization
	w = (Z'*Z + eye(size(Z,2))*l)\(Z'*y)

	display(size(w))

	# Make linear prediction function
	predict(Xtilde) = rbfBasis(Xtilde, X, sigma)*w

	# Return model
	return LinearModel(predict,w)
end
