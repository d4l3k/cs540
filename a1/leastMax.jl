using MathProgBase, GLPKMathProgInterface

include("misc.jl")

function leastMax(X, y)
  (n, d) = size(X)
  X = [ones(n, 1) X]
  d += 1
  y = y[:,1]
  c = [ones(n); zeros(d)]
  A = zeros(0, d + n)
  b = Float64[]

  for j = 1:d
    colj = zeros(d, d)
    colj[j,j] = 1
    A = [A; eye(n) X*colj; eye(n) -X*colj]
    b = [b; -y; y]
  end

  display(size(y))
  display(size(c))
  display(size(A))
  display(size(b))

  l = [zeros(n); ones(d) * -Inf]

  sol = linprog(c, A, '>', b, l, Inf, GLPKSolverLP())
  w = sol.sol[n+1:n+d]
  if sol.status == :Optimal
      println("Optimal objective value is $(sol.objval)")
      println("Optimal solution vector is: $(w)")
  else
      println("Error: solution status $(sol.status)")
  end
  predict(Xtilde) = [ones(size(Xtilde,1),1) Xtilde]*w

  # Return model
  return LinearModel(predict, w)
end
