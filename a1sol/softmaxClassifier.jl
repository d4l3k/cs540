include("misc.jl")
include("findMin.jl")

function softmaxObj(w,X,y, k)
        (n, d) = size(X)
        f = 0
        for i = 1:n
          b = 0
          for c = 1:k
            b += exp(w[:, c]' * X[i, :])
          end
          f += -w[:, y[i]]' * X[i, :] + log(b)
        end
	g = zeros(d,k)
        for i = 1:n
          b = 0
          for c = 1:k
            b += exp(w[:, c]' * X[i, :])
          end
          for c = 1:k
            if y[i] == c
	      g[:,c] += -X[i, :]
            end
            g[:,c] += 1/b * exp(w[:, c]' * X[i, :]) * X[i, :]
          end
        end
        #a = -w[:, y]*X
        #b = w[:, ones(k, size(y,2)) * (1:Int(k))']*X
	#f = sum(a + log(b))
	display(f)
	display(g)
	return (f,g)
end

# Multi-class softmax classifier
function softmaxClassifier(X,y)
	(n,d) = size(X)
	k = maximum(y)


	W = zeros(d,k)

	display(size(W))

	# Each binary objective has the same features but different lables
	funObj(w) = softmaxObj(w,X,y, k)

	W = findMin(funObj, W, verbose=true)#, derivativeCheck=true)

	# Make linear prediction function
	#predict(Xhat) = findmax(Xhat*W)[2]
	predict(Xhat) = mapslices(indmax,Xhat*W,2)

	return LinearModel(predict,W)
end
