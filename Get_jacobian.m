function Jacobian = Get_jacobian(f_function,x,SpPat)  
  %-----------------------------------------------------------------------------
  % Get_jacobian:  Uses the forward mode of automatic differentiation (Deriv)
  %                to compute the gradient of a vector valued function f(x) with 
  %                respect to components of the variable x.
  %
  %                This method provides accuracy though may be computationally 
  %                expensive.  If a directional derivative is required, in
  %                other words, a jacobian*vector product, a separate
  %                function should be created for this.
  %
  %                This function provides optional savings through sparse 
  %                jacobian information implemented by Gene Cliff, 2013.
  %                See  A.R.Curtis, M.J.D.Powell, and K.J.Reid,
  %                     'On Estimation of Sparse Jacobian Matrices', 
  %                     Journal of the Institute for Mathematics and its 
  %                     Applications, 1974, 13, 117-119.
  %                for implementation details.
  %
  %  Usage:        [ Jacobian ] = Get_jacobian( f_function, x, SparsPattern )
  %
  %  Variables:
  %                f_function   - a function handle (to an m-vector function)
  %                x            - a vector of length n
  %                SparsPattern - a matrix of ones and zeros indicating
  %                               nonzero entries in the Jacobian
  %                               (optional)
  %
  %                Jacobian     - a matrix of dimension m x n
  %
  %  Authors:      Jeff Borggaard and Gene Cliff, 2013
  %
  %  License:      LGPL 3.0
  %% ---------------------------------------------------------------------------
  
  n_var = length(x);
  x=x(:);
  
  if ( nargin==2 )
    Jacobian = [];%zeros(n_var,1);
  
    for n=1:n_var
      a = Set_variable(x(n));
    
      c = [ x(1:n-1); a; x(n+1:n_var)];

      f = feval(f_function,c);
      Jacobian(:,n) = Get_deriv(f);      %#ok
    end
  
  else
    m_fun   = size(SpPat,1);
    x_fd    = dx_sparsity(SpPat);
    num_inc = length(x_fd);         % the number of independent increments
    
    %k       = nnz(SpPat);           % allocate storage for the sparse Jacobian
    II      = [];   %zeros(1,k); 
    JJ      = [];   %zeros(1,k);
    XX      = [];   %zeros(1,k);
    
    o       = ones(n_var,1);
    
    for kk = 1:num_inc
      n_loc     = x_fd{kk}; % list of independent variables in this group
      dx        = zeros(n_var,1);
      dx(n_loc) = o(n_loc);
      xd       = Deriv(x,dx);
      f        = feval(f_function,xd);
      df       = Get_deriv(f);
      
      for jj=n_loc
        ii = find( SpPat(:,jj)~=0 );
        II = [II; ii];                     %#ok
        JJ = [JJ; jj*ones(length(ii), 1)]; %#ok
        XX = [XX; df(ii)];                 %#ok
      end
    end
    
    Jacobian = sparse(II, JJ, XX, m_fun, n_var);

  end
end
    

function  x_fd  = dx_sparsity(J)
% Assemble groups of variables for finite difference increments
% based on the sparsity pattern J
% Ref: A.R.Curtis, M.J.D.Powell, and K.J.Reid,
% 'On Estimation of Sparse Jacobian Matrices', 
% J. of the Institute for Mathematics and Its Applicat, 1974, 13, 117-119.

% We wish to compute the Jacobian, J =  {\partial f}{\partial x} m \times n
% x_fd is a cell array, x_fd{i} enumerates elements of x that can be
% simultaneously incremented
%
   nc    = size(J, 2);
 
   x_fd  = cell(floor(nc/2),1); % pre-allocate an estimated size  
   col   = 1:nc;
   i_fd  = 0;
   
   while ~isempty(col)
       i_fd       = i_fd + 1;
       x_fd{i_fd} = locate(col, J);%assemble variables lists
       col        = setdiff(col, x_fd{i_fd}); % decrement the list of cols
   end
   
   x_fd = x_fd(1:i_fd,1); % trim any empty cell arrays
       
end
%  sub function
   function xd_j = locate(col_remain, J)
% Given the sparsity pattern (J) and a list of remaining columns, assemble
% a list of columns (variables) for a sparse finite-difference. A variable
% is added to the list iff it does not enter any function already affected

      xd_j = [];                     % initialize to an empty
      jc   = zeros(size(J(:,1)));    % no functions yet affected
    
      for jj=col_remain
          if ~any( jc & J(:,jj) )    % any previous functions affected
              xd_j = [xd_j jj]; %#ok % add this variable to the list
              jc   = jc | J(:, jj);  % update the functions affected
          end
      end
              
   end

