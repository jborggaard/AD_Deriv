function obj = Set_variable(c)  
% a less confusing, but less flexible way to define the independent
% variable for differentiation
  
  obj = Deriv(c,1);
  
end
    

