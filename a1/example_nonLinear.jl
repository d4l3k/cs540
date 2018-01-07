# Load X and y variable
using JLD
data = load("nonLinear.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])

# Compute number of training examples and number of features
(n,d) = size(X)

display(size(X))

Xtrain = X[1:n/2]
ytrain = y[1:n/2]
Xvalidate = X[n/2:n]
yvalidate = y[n/2:n]

# Fit least squares model
#include("leastSquares.jl")
#model = leastSquares(X,y)


bestError = 10000000000000000
bestl = 0
bestsigma = 0

include("leastSquaresRBFL2.jl")

# Find best l, sigma values
for l = 0.2:0.4:2
  for sigma = 0.2:0.4:2
		model = leastSquaresRBFL2(Xtrain,ytrain,1,1)

		# Report the error on the validation set
		t = size(Xvalidate,1)
		yhat = model.predict(Xvalidate)
		validationError = sum((yhat - yvalidate).^2)/t
		@printf("l = %f, sigma = %f, ValidationError = %.2f\n", l, sigma, validationError)
    if validationError < bestError
      bestError = validationError
      bestl = l
      bestsigma = sigma
    end
  end
end

@printf("Best validation l = %f, sigma = %f, error = %f", bestl, bestsigma, bestError)

model = leastSquaresRBFL2(X,y,bestl,bestsigma)

# Report the error on the test set
t = size(Xtest,1)
yhat = model.predict(Xtest)
testError = sum((yhat - ytest).^2)/t
@printf("l = %f, sigma = %f, TestError = %.2f\n", l, sigma, testError)


# Plot model
using PyPlot
figure()
plot(X,y,"b.")
plot(Xtest,ytest,"g.")
Xhat = minimum(X):.1:maximum(X)
Xhat = reshape(Xhat,length(Xhat),1) # Make into an n by 1 matrix
yhat = model.predict(Xhat)
plot(Xhat,yhat,"r")
ylim((-300,400))
show()
