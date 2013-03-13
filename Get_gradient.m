function gradient = Get_gradient(f_function,x)  
% This file uses the forward mode of automatic differentiation to compute
% the gradient of a function with respect to components of the variable x.
% This method provides accuracy, even though it does not provide
% significant computational savings.  For the latter, the reverse mode of
% automatic differentiation should be used (currently not implemented).
  
  n_var = length(x);
  x=x(:);
  
  gradient = zeros(n_var,1);
  
  for n=1:n_var
    a = Set_variable(x(n));
    
    c = [ x(1:n-1); a; x(n+1:n_var)];

    f = feval(f_function,c);
    gradient(n) = Get_deriv(f);
  end
  
end
    