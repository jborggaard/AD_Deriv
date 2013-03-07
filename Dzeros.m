function D = Dzeros(a,b)
%% This function is used to overcome the preallocation problem.  When the
%  following commands are executed, 
%
%  > c = zeros(3,3);
%  > c(1,1) = Deriv(1,1);
%
%  there is an error.  Since c was originally defined as a double object this
%  command cannot promote the variable to a Deriv object.  The workaround
%  is to perform a global find/replace of "zeros" to "Dzeros" in the
%  differentiated code.  While that overcomes the problem, a better solution 
%  is to apply this change only to those arrays that should be promoted to
%  Deriv objects.
%%
  if ( nargin==1 )
    D = Deriv( zeros(a), zeros(a) );
  else
    D = Deriv( zeros(a,b), zeros(a,b) );
  end
end