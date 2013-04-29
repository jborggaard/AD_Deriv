function gradient = Get_gradient(f_function,x)  
  %-----------------------------------------------------------------------------
  % Get_gradient:  Uses the forward mode of automatic differentiation (Deriv)
  %                to compute the gradient of a function f(x) with respect to 
  %                components of the variable x.
  %
  %                This method provides accuracy, even though it does not 
  %                provide significant computational savings.  For the latter, 
  %                the reverse mode of automatic differentiation should be used,
  %                when implemented in Deriv.
  %
  %  Usage:        [ gradient ] = Get_gradient( f_function, x )
  %
  %  Variables:
  %                f_function  - a function handle
  %                x           - a vector of length n
  %
  %                gradient    - a vector of length n
  %
  %  Author:       Jeff Borggaard, 2013
  %
  %  License:      LGPL 3.0
  %% ---------------------------------------------------------------------------
  
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
    