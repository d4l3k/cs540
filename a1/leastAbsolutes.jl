using MathProgBase, Clp

include("misc.jl")

function leastAbsolutes(X, y)
  (n, d) = size(X)
  d += 1
  y = y[:,1]
  c = [ones(n); zeros(d)]
  A = [eye(n) ones(n, 1) -X; eye(n) ones(n, 1) X]'
  b = [-y; y]

  display(size(y))
  display(size(c))
  display(size(A))
  display(size(b))

  sol = linprog(c, A, '>', b, ClpSolver())
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
