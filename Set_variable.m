function obj = Set_variable(c)  
% a less confusing, but less flexible way to define an independent, scalar
% variable for differentiation
  
  obj = Deriv(c,1);
  
end
    

