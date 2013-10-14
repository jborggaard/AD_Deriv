classdef Deriv
  %-----------------------------------------------------------------------------
  %   The Deriv class is a value class that implements automatic differentiation
  %   using operator overloading.  The distinctive feature of this 
  %   implementation is that it includes a number of matrix factorizations that 
  %   are useful in POD and solving control problems and use of continuous
  %   sensitivity equations to differentiate adaptive algorithms and other
  %   algorithms without a consistent/discrete derivative.  Removing step
  %   size dependence/issues with nondifferentiable functions will make this
  %   AD more transparent.
  %
  %   Some of these functions rely on the control toolbox.
  %
  %   Currently we have only implemented the forward mode for now.
  %
  %   Author:  Jeff Borggaard, 2012
  %            Virginia Tech
  %
  %   License:  This code is distributed under the Gnu LGPL license.
  %
  %   References include:
  %   
  %   Griewank, Computing Derivatives, SIAM
  %     Gives a general overview of automatic differentiation, including the
  %     operator overloading approach.
  %   Coleman and Verma
  %     The first instance of using Matlab classes to implement AD in Matlab.
  %   Borggaard and Verma
  %     The first combination of discrete and continuous sensitivity analysis
  %     using automatic differentiation.
  %   Forth,
  %     Included more functionality and integrated it with an optimization
  %     package.
  %   MATLAB Reference Manual, 2012b
  %-----------------------------------------------------------------------------
  
  properties
    x        % think:   "object"   (x)
    dx       % think:  d"object"/dx(x)
  end
  
  % Class methods
  methods
    function obj = Deriv(c,d)
      % Usage: Deriv(c,d) constructs a Deriv object: 
      %   "c" is the value associated with c, and 
      %   "d" is the derivative of c with respect to the vector of "parameters"
      if ( nargin==0 )
        obj.x  = 0;
        obj.dx = 0;
      elseif ( isempty(c) )
        obj.x  = [];
        obj.dx = [];
      elseif ( isa(c,'Deriv') )
        obj = c;%struct('x',c.x,  'dx',c.dx);
      elseif ( nargin==1 )
        obj.x  = c;
        obj.dx = 0*c;
      else
        if ( size(c)==size(d) )
          obj.x  = c;
          obj.dx = d;
        else
          error('array of values and derivatives are not compatible')
        end
      end
    end % Deriv constructor
    
    %---------------------------------------------------------------------------
    % Embedding Useful Functionality
    %---------------------------------------------------------------------------
    function out = Get_value(c)  
      if ( isa(c,'Deriv') )
        out = c.x;
      else
        out = c;
      end
    end
    
    function out = Get_deriv(c)
      out = c.dx;
    end
     
    
    % Parses multiple vargin inputs that are of type Deriv but only seen as a
    % cell (bypass must be a Deriv for internal functions to find this).
    function [x,dx] = process_varargin(obj,bypass)                          %#ok
      x  = cell(size(obj));
      dx = cell(size(obj));
      for i=1:length(x)
        if ( isa(obj{i},'Deriv') )
          x{i}  = obj{i}.x;
          dx{i} = obj{i}.dx;
        else
          x{i}  = obj{i};
          dx{i} = zeros(size(obj{i}));
        end
      end
    end
    

    %---------------------------------------------------------------------------
    % From this point, we overload Matlab functions and operators in the
    % order they appear in the Mathematics portion of the documentation:
    %   MATLAB Function Reference R2012b: 
    %---------------------------------------------------------------------------
    % Basic Information p.1-32
    %% -------------------------------------------------------------------------
    function [] = disp(c)
      fprintf('\n')
      fprintf(' value: \n'); disp(c.x )
      fprintf(' deriv: \n'); disp(c.dx)
      fprintf('\n')
    end
    
    function [] = display(c)
      if isequal(get(0,'FormatSpacing'),'compact')
        disp([inputname(1) ' ='])
        disp(c.x)
      else
        disp(' ')
        disp([inputname(1) ' ='])
        disp(' ')
        disp(c.x)
      end
    end
    
    function [] = fprintf(varargin)
      bypass = Deriv( 1, 1 );
      [X,DX] = process_varargin(varargin,bypass);
      
      if ( ischar(X{1}) )
        fprintf(X{1},X{2:end});
      else
        fprintf(X{1},X{2},X{3:end});
      end
    end
    
    
    function bool = iscolumn(c)
      bool = iscolumn(c.x);
    end
    
    function bool = isempty(c)
      bool = isempty(c.x);
    end
    
    function bool = isequal(c,d)
      if ( ~isa(c,'Deriv') )
        bool = isequal(c,d.x);
      elseif ( ~isa(d,'Deriv') )
        bool = isequal(c.x,d);
      else
        bool = isequal(c.x,d.x);
      end
    end
    
    function bool = isequaln(c,d)
      if ( ~isa(c,'Deriv') )
        bool = isequaln(c,d.x);
      elseif ( ~isa(d,'Deriv') )
        bool = isequaln(c.x,d);
      else
        bool = isequaln(c.x,d.x);
      end
    end
    
    function bool = isfinite(c)
      bool = isfinite(c.x);
    end
    
    function bool = isfloat(c)
      bool = isfloat(c.x);
    end
    
    function bool = isinf(c)
      bool = isinf(c.x);
    end
    
    function bool = isinteger(c)
      bool = isinteger(c.x);
    end
    
    function bool = islogical(c)
      bool = islogical(c.x);
    end
    
    function bool = ismatrix(c)
      bool = ismatrix(c.x);
    end
    
    function bool = isnan(c)
      bool = isnan(c.x);
    end
    
    function bool = isnumeric(c)
      bool = isnumeric(c.x);
    end
    
    function bool = isrow(c)
      bool = isrow(c.x);
    end
    
    function bool = isscalar(c)
      bool = isscalar(c.x);
    end
    
    function bool = issparse(c)
      bool = issparse(c.x);
    end
    
    function bool = isvector(c)
      bool = isvector(c.x);
    end
    
    function [m] = length(c)
      m = length(c.x);
    end
    
    function [m,i] = max(c,d,e)
      % max is not always differentiable, but often is      
      if ( nargin==1 )
        [nr,nc] = size(c.x);

        n = min(nr,nc);
        if ( n==1 && nargin==1 )  % vector case
          [y,i] = max(c.x);
          m     = Deriv( y, c.dx(i) );
        elseif ( nargin==1 )      % matrix case
          [y,i] = max(c.x);
          m     = Deriv( y, diag(c.dx(i,:))' );
        end
      elseif ( ~isempty(d) )
        if ( ~isa(c,'Deriv') )
          c = Deriv(c,zeros(size(c)));
        end
        if ( ~isa(d,'Deriv') )
          d = Deriv(d,zeros(size(d)));
        end
      
        if ( isscalar(c.x) )
          [nr,nc] = size(d.x);
          c = Deriv(c.x*ones(nr,nc),c.dx*ones(nr,nc));
        end
        if ( isscalar(d.x) )
          [nr,nc] = size(c.x);
          d = Deriv(d.x*ones(nr,nc),d.dx*ones(nr,nc));
        end
      
        [nr,nc] = size(c.x);

        n = min(nr,nc);
        if ( n==1 && nargin==1 )  % vector case
          [y,i] = max(c.x);
          m     = Deriv( y, c.dx(i) );
        elseif ( nargin==1 )      % matrix case
          [y,i] = max(c.x);
          m     = Deriv( y, diag(c.dx(i,:))' );
        elseif ( sum(size(d.x))>0 ) % the comparison case
        
          if ( nargout>1 )
            error('two output arguments not supported for max with two matrices');
          end
        
          if ( size(c.x) ~= size(d.x) )
            error('incompatible sizes\n')
          end
        
          y  = zeros(nr,nc);  % preallocate storage
          yd = zeros(nr,nc);
        
          for i=1:nr
            for j=1:nc
              if ( c.x(i,j) > d.x(i,j) )
                y(i,j)  = c.x(i,j);
                yd(i,j) = c.dx(i,j);
              elseif ( d.x(i,j) > c.x(i,j) )
                y(i,j)  = d.x(i,j);
                yd(i,j) = d.dx(i,j);
              else
                y(i,j)  = c.x(i,j);
                yd(i,j) = ( c.dx(i,j) + d.dx(i,j) )/2;
                warning('derivative not uniquely defined for entry (%d,%d)\nthe average value is used',i,j)
              end
            end
          end
         
          m = Deriv( y, yd );
       
        end
       
      elseif ( e==1 )
        [y,i] = max(c.x,[],1);
        m     = Deriv(y, diag(c.dx(i,:))');
        
      elseif ( e==2 )
        [y,i] = max(c.x,[],2);
        m     = Deriv(y, diag(c.dx(:,i)));
        
      else
        error('max is only defined for two dimensional arrays')
      end
         
    end
    
    function [m,i] = min(c,d,e)
      % min is not always differentiable, but often is          
      if ( nargin==1 )
        [nr,nc] = size(c.x);

        n = min(nr,nc);
        if ( n==1 && nargin==1 )  % vector case
          [y,i] = max(c.x);
          m     = Deriv( y, c.dx(i) );
        elseif ( nargin==1 )      % matrix case
          [y,i] = max(c.x);
          m     = Deriv( y, diag(c.dx(i,:))' );
        end
      elseif  ( ~isempty(d) )
        if ( ~isa(c,'Deriv') )
          c = Deriv(c,zeros(size(c)));
        end
        if ( ~isa(d,'Deriv') )
          d = Deriv(d,zeros(size(d)));
        end
      
        if ( isscalar(c.x) )
          [nr,nc] = size(d.x);
          c = Deriv(c.x*ones(nr,nc),c.dx*ones(nr,nc));
        end
        if ( isscalar(d.x) )
          [nr,nc] = size(c.x);
          d = Deriv(d.x*ones(nr,nc),d.dx*ones(nr,nc));
        end
      
        [nr,nc] = size(c.x);

        n = min(nr,nc);
        if ( n==1 && nargin==1 )  % vector case
          [y,i] = min(c.x);
          m     = Deriv( y, c.dx(i) );
        elseif ( nargin==1 )      % matrix case
          [y,i] = min(c.x);
          m     = Deriv( y, diag(c.dx(i,:))' );
        elseif ( sum(size(d.x))>0 ) % the comparison case
        
          if ( nargout>1 )
            error('two output arguments not supported for min with two matrices');
          end
        
          if ( size(c.x) ~= size(d.x) )
            error('incompatible sizes\n')
          end
        
          y  = zeros(nr,nc);  % preallocate storage
          yd = zeros(nr,nc);
        
          for i=1:nr
            for j=1:nc
              if ( c.x(i,j) < d.x(i,j) )
                y(i,j)  = c.x(i,j);
                yd(i,j) = c.dx(i,j);
              elseif ( d.x(i,j) < c.x(i,j) )
                y(i,j)  = d.x(i,j);
                yd(i,j) = d.dx(i,j);
              else
                y(i,j)  = c.x(i,j);
                yd(i,j) = ( c.dx(i,j) + d.dx(i,j) )/2;
                warning('derivative not uniquely defined for entry (%d,%d)\nthe average value is used',i,j)
              end
            end
          end
         
          m = Deriv( y, yd );
       
        end
        
      elseif ( e==1 )
        [y,i] = min(c.x,[],1);
        m     = Deriv(y, diag(c.dx(i,:))');
        
      elseif ( e==2 )
        [y,i] = min(c.x,[],2);
        m     = Deriv(y, diag(c.dx(:,i)));
        
      else
        error('min is only defined for two dimensional arrays')
      end
          
    end
    
    function [m] = ndims(c)
      m = ndims(c.x);
    end
    
    function [m] = numel(c)
      m = numel(c.x);
    end
    
    function [varargout] = size(c,n)
      if ( nargout==1 && nargin==2 )
        out = size(c.x,n);
        varargout = {out};
      elseif ( nargout>1 && nargin==2 )
        error('Too many output arguments.')
      else
        out = size(c.x);
           
        if ( nargout==1 )
          varargout = {out};
          
        elseif ( nargout>1 )
          m = length(out);
          varargout = cell(1,m);
          for i=1:m
            varargout{i} = out(i);
          end
        end
      end
    end
    %%  

    %---------------------------------------------------------------------------
    % Operators p.1-33
    %% -------------------------------------------------------------------------
    function obj = ctranspose(c)
      obj = Deriv( c.x', c.dx' );  
    end
    
    function obj = minus(c,d)
      if ( ~isa(c,'Deriv') )                       % c is a scalar, d is a Deriv
        obj = Deriv( c - d.x, - d.dx );
      elseif ( ~isa(d,'Deriv') )                   % d is a scalar, c is a Deriv
        obj = Deriv( c.x - d, c.dx );
      else
        obj = Deriv( c.x - d.x, c.dx - d.dx );
      end
    end
       
    function obj = mpower(c,d)
      if ( ~isa(d,'Deriv') )
        obj = Deriv( c.x.^d, d.*c.x.^(d-1).*c.dx );
      elseif ( ~isa(c,'Deriv') )
        obj = Deriv( c.^d.x, c.^d.x .*d.dx * log(c) );
      else
        obj = Deriv( c.x.^d.x, d.x.*c.x.^(d.x-1).*c.dx ... 
                              +c.x.^d.x .*d.dx * log(c.x) );
      end
    end
    
    function obj = plus(c,d)
      if ( ~isa(c,'Deriv') )                       % c is a scalar, d is a Deriv
        obj = Deriv( c + d.x, d.dx.*ones(size(c)) );
      elseif ( ~isa(d,'Deriv') )                   % d is a scalar, c is a Deriv
        obj = Deriv( c.x + d, c.dx.*ones(size(d)) );
      else
        obj = Deriv( c.x  + d.x, c.dx + d.dx );
      end
    end
    
    function obj = times(c,d)    % c.*d
      if ( ~isa(c,'Deriv') )                       % c is a scalar, d is a Deriv
        obj = Deriv( c.*d.x, c.*d.dx );
      elseif ( ~isa(d,'Deriv') )                   % d is a scalar, c is a Deriv
        obj = Deriv( c.x .*d, c.dx.*d );
      else
        obj = Deriv( c.x.*d.x, c.x.*d.dx + c.dx.*d.x );
      end
    end
    
    function obj = transpose(c)
      obj = Deriv( transpose(c.x), transpose(c.dx) );  
    end
    
    function obj = uminus(c)
      obj = Deriv( -c.x, -c.dx );  
    end
    
    function obj = uplus(c)
      obj = c;
    end
    %%  

    %---------------------------------------------------------------------------
    % Elementary Matrices and Arrays p.1-34 (not all listed functions are
    % overloaded since they don't make sense to differentiate, including
    % rand, randi, randn, RandStream, rng)
    %% -------------------------------------------------------------------------
    function obj = blkdiag(varargin)
      bypass = Deriv( 1, 1 );
      [X,DX] = process_varargin(varargin,bypass);
      
      obj = Deriv( blkdiag(X{:}), blkdiag(DX{:}) ); 
    end
    
    function obj = diag(v,k)
      if ( nargin==1 )
        obj = Deriv( diag(v.x  ), diag(v.dx  ) );
      else
        obj = Deriv( diag(v.x,k), diag(v.dx,k) );
      end
    end
    
    function obj = Deye(m,n)  % should sometimes replace eye
      % Perhaps there is a way to eventually call this instead of a
      % preallocated eye.
      obj = Deriv( eye(m,n), zeros(m,n) );
    end
    
    function obj = freqspace(a,b,m)
      error('function freqspace not implemented')
    end
    
    function out = ind2sub(c,m)
      error('function ind2sub not implemented')
    end
    
    function obj = linspace(a,b,m)
      if ( nargin==2 )
        m = 100;
      end
      
      if ( ~isa(a,'Deriv') )
        a = Deriv( a, zeros(size(a)) );
      elseif ( ~isa(b,'Deriv') )
        b = Deriv( b, zeros(size(b)) );
      end
      
      obj = Deriv( linspace(a.x,b.x,m), linspace(a.dx,b.dx,m) );
    end
    
    function obj = logspace(a,b,m)
      if ( nargin==2 )
        m = 50;
      end
      
      if ( ~isa(a,'Deriv') )
        a = Deriv( a, zeros(size(a)) );
      elseif ( ~isa(b,'Deriv') )
        b = Deriv( b, zeros(size(b)) );
      end
      
      obj = Deriv( logspace(a.x,b.x,m), logspace(a.dx,b.dx,m) );
    end
    
    % ndgrid (below under Domain Generation)

    function obj = Dones(m,n)  % should sometimes replace ones
      % Perhaps there is a way to eventually call this instead of a
      % preallocated ones.  For now, this is a separate function in the same
      % path as Deriv.
      obj = Deriv( ones(m,n), zeros(m,n) );
    end
    
    function out = rand(m,n)
      error('function rand not implemented')
    end
    
    function out = randi(m,n)
      error('function randi not implemented')
    end
    
    function out = randn(m,n)
      error('function randn not implemented')
    end
    
    function out = rng(m,n)
      error('function rng not implemented')
    end
    
    function out = sub2ind(c,m)
      error('function sub2ind not implemented')
    end
    
    function obj = Dzeros(m,n)  % should sometimes replace zeros
      % Perhaps there is a way to eventually call this instead of a
      % preallocated zeros.  For now, this is a separate function in the same
      % path as Deriv.  A preprocessing step could be implemented to
      % automatically replace zeros by Dzeros to make this more user
      % transparent.
      obj = Deriv( zeros(m,n), zeros(m,n) );
    end
    %%

    %---------------------------------------------------------------------------
    % Array Operations p.1-35
    %% -------------------------------------------------------------------------
    function obj = accumarray(c,v,sz,fun,fillval,is_sparse)
      error('function accumarray not yet implemented')
    end
    
    function obj = arrayfun(func,varargin)
      error('function arrayfun not yet implemented')
    end
    
    function obj = bsxfun(fun,c,d)
      error('function bsxfun not yet implemented')
    end
    
    % cast -> can't change from a Deriv
    
    function obj = cross(c,d,dim)
      if ( nargin==2 )
        obj = Deriv( cross(c.x,d.x    ), cross(c.dx,d.x    )+cross(c.x,d.dx    ) );
      else
        obj = Deriv( cross(c.x,d.x,dim), cross(c.dx,d.x,dim)+cross(c.x,d.dx,dim) );
      end
    end
    
    function obj = cumprod(c,m)
      [nr,nc] = size(c.x);
      
      if (nr==1 || nc==1)   % vector case
        px = cumprod(c.x);
        pdx = zeros(size(c.x));
        pdx(1) = c.dx(1);
        
        for j=2:max(nr,nc)
          pdx(j) = pdx(j-1)*c.x(j) + px(j-1)*c.dx(j);
        end
        
      elseif ( nargin==1 || m==1 )
        px = cumprod(c.x);
        pdx = zeros(size(c.x));
        pdx(1,:) = c.dx(1,:);
        
        for j=2:nr
          pdx(j,:) = pdx(j-1,:).*c.x(j,:) + px(j-1,:).*c.dx(j,:);
        end
        
      else
        px = cumprod(c.x,2);
        pdx = zeros(size(c.x));
        pdx(:,1) = c.dx(:,1);
        
        for j=2:nc
          pdx(:,j) = pdx(:,j-1).*c.x(:,j) + px(:,j-1).*c.dx(:,j);
        end
      end
      
      obj = Deriv( px, pdx );
        
    end
    
    function obj = cumsum(c,m)
      if ( nargin==1 )
        obj = Deriv( cumsum(c.x  ), cumsum(c.dx  ) );
      else
        obj = Deriv( cumsum(c.x,m), cumsum(c.dx,m) );
      end
    end
        
    function obj = dot(c,d,dim)
      if ( nargin==2 )
        obj = Deriv( dot(c.x,d.x    ), dot(c.dx,d.x    )+dot(c.x,d.dx    ) );
      else
        obj = Deriv( dot(c.x,d.x,dim), dot(c.dx,d.x,dim)+dot(c.x,d.dx,dim) );
      end
    end
    
    function out = idivide(c,d,opt)
      % doesn't make sense for c or d to be a Deriv object, demote the result
      c = Deriv(c);
      d = Deriv(d);
      if ( nargin==2 )
        out = idivide(c.x,d.x);
      else
        out = idivide(c.x,d.x,opt);
      end
      warning('attempted to differentiate the function idivide')
    end
    
    function obj = prod(c,dim)
      if ( nargin==1 )
        px  = prod(c.x);
        % find first non-singleton dimension
        dim = find( size(c.x)-1 ); 
        if ( isempty(dim) )
          dim = 1;
        end
      else
        px  = prod(c.x,dim); 
      end
      
      pdx = zeros(size(px));
      n   = size(c.x,dim);
      
      % only works for 1-dimensional arrays
      if ( px~=0 )
        for i=1:n
          pdx = pdx + px*c.dx(i)/c.x(i);
        end
        
      else
        izero = find( c.x==0 );
        if ( length(izero)==1 )
          index = [ 1:izero-1, izero+1:n ];
          pdx = c.dx(izero)*prod(c.x(index));
        else
          % leave pdx=0
        end
      end
      
      obj = Deriv( px, pdx );
    end
    
    function obj = sum(c,d)
      if ( nargin==1 )
        obj = Deriv( sum(c.x), sum(c.dx) );
      else
        obj = Deriv( sum(c.x,d), sum(c.dx,d) );
      end
    end
    
    function obj = surfnorm(x,y,z)
      error('function surfnorm not yet implemented')
    end
    
    function obj = tril(c,k)
      if ( nargin==1 )
        obj = Deriv( tril(c.x), tril(c.dx) );
      else
        obj = Deriv( tril(c.x,k), tril(c.dx,k) );
      end
    end
    
    function obj = triu(c,k)
      if ( nargin==1 )
        obj = Deriv( triu(c.x), triu(c.dx) );
      else
        obj = Deriv( triu(c.x,k), triu(c.dx,k) );
      end
    end
    %%
    
    %---------------------------------------------------------------------------
    % Array Manipulation p.1-36  (skipping functions defined above)
    %% -------------------------------------------------------------------------
    % blkdiag (above)
    
    function obj = cat(dim,varargin)
      bypass = Deriv( 1, 1 );
      [X,DX] = process_varargin(varargin,bypass);
      obj = Deriv( cat(dim,X{:}), cat(dim,DX{:}) );
    end
    
    function obj = circshift(c,shiftsize)
      if ( nargin==1 )
        obj = Deriv( circshift(c.x          ), circshift(c.dx          ) );
      else
        if ( isa(shiftsize,'Deriv') )
          shiftsize = shiftsize.x;
        end
        obj = Deriv( circshift(c.x,shiftsize), circshift(c.dx,shiftsize) );
      end
    end
    
    % diag (above)
    
    function out = end(a,b,c)
      if ( nargin==1 )
        out = builtin('end',a.x);
      elseif ( nargin==2 )
        out = builtin('end',a.x,b);
      elseif ( nargin==3 )
        out = builtin('end',a.x,b,c);
      end
    end
       
    function obj = flipdim(c,n)
      obj = Deriv( flipdim(c.x,n), flipdim(c.dx,n) );
    end
    
    function obj = fliplr(c)
      obj = Deriv( fliplr(c.x), fliplr(c.dx) );
    end
    
    function obj = flipud(c)
      obj = Deriv( flipud(c.x), flipud(c.dx) );
    end
    
    function obj = horzcat(varargin)
      bypass = Deriv( 1, 1 );
      [X,DX] = process_varargin(varargin,bypass);
      obj = Deriv( horzcat(X{:}), horzcat(DX{:}) );
    end
    
    % inline
    
    function obj = ipermute(c,order)
      if ( ~isa(order,'Deriv') )
        order = order.x;
      end
      
      obj = Deriv( ipermute(c.x,order), ipermute(c.dx,order) );  
    end
    
    
    function obj = permute(c,order)
      if ( ~isa(order,'Deriv') )
        order = order.x;
      end
      
      obj = Deriv( permute(c.x,order), permute(c.dx,order) );  
    end
    
    % repmat
    
    function obj = reshape(c,m,n)
      obj = Deriv( reshape(c.x,m,n), reshape(c.dx,m,n) );
    end
    
    function obj = rot90(c,k)
      if ( nargin==1 )  
        obj = Deriv( rot90(c.x), rot90(c.dx) );
      else
        obj = Deriv( rot90(c.x,k), rot90(c.dx,k) );
      end
    end
    
    % shiftdim
    
    % sort
    
    % sortrows
    
    function obj = squeeze(c)
      obj = Deriv( squeeze(c.x), squeeze(c.dx) );
    end
    
    % vectorize
    
    function obj = vertcat(varargin)
      bypass = Deriv( 1, 1 );
      [X,DX] = process_varargin(varargin,bypass);
      obj = Deriv( vertcat(X{:}), vertcat(DX{:}) );
    end
    
    %%  
    
    %---------------------------------------------------------------------------
    % Specialized Matrices p.1-37  (most functions in this section are
    % ignored)
    %% -------------------------------------------------------------------------
    function obj = compan(c)
      cx  = c.x(:)';
      cdx = c.dx(:)';
      
      Ax  = compan(cx);
      Adx = zeros(size(Ax));
      Adx(1,:) = -cdx(2:end)/cx(1) + cx(2:end)*cdx(1)/cx(1)^2;
      obj = Deriv( Ax, Adx );
    end
    
    % gallery, hadamard, hankel, hilb, invhilb, magic, pascal, rosser
    
    function obj = toeplitz(c,r)
      if ( nargin==1 )
        obj = Deriv( toeplitz(c.x), toeplitz(c.dx) );
      else % may have issues with complex values, but for real, it is
        obj = Deriv( toeplitz(c.x,r.x), toeplitz(c.dx,r.dx) );
      end
    end
    
    function obj = vander(c)
      n   = length(c.x);
      if ( n==1 )
        obj = Deriv( 1, 0 );
      else
        Vdx        = zeros(n,n);
        Vdx(:,n-1) = c.dx;
        
        for i=n-2:-1:1
          Vdx(:,i) = c.x.*Vdx(:,i+1);
        end
        Vdx = Vdx*diag(n-1:-1:0);
        
        obj = Deriv( vander(c.x), Vdx );
      end
    end
    
    % wilkinson
    %%

    % Linear Algebra
    %---------------------------------------------------------------------------
    % Matrix Analysis p.1-38
    %% -------------------------------------------------------------------------
    % cond, condeig, det
    
    function obj = norm(c,p)      
      if ( nargin==1 ) % default is 2-norm
        p = 2;
      end
      
      v = size(c.x);
      if ( length(v)>2 )
        error('norm is not implemented for high dimensional arrays.');
      end
      
      if ( v(1)==1 || v(2)==1 )  
        %-----------------------------------------------------------------------
        % vector norms
        %-----------------------------------------------------------------------
        if ( p==1 )                                                      % p = 1
          obj = Deriv( norm(c.x,1), sum( sign(c.x).*c.dx ) );
            
        elseif ( p==2 )                                                  % p = 2
          nc = norm(c.x,2);
          if ( nc~=0 )
            obj = Deriv( nc, sum( c.x.*c.dx )/nc );
          else
            obj = Deriv( nc, sign( sum(c.x.*c.dx)*Inf ) );
          end
          
        elseif ( isinf(p) )
          % The standard infinity norm of a vector.  Note that this
          % function is consistent with Matlabs accommodation of a "-Inf"
          % norm.
          %
          % To accomodate the special case where two or more elements are 
          % tied for largest absolute value, we use the sort function 
          % instead of the max function.  This implements the right 
          % Dini derivative
 
          % still need to sort out complex c.dx and abs function
          
          if ( p>0 )                                              % p = infinity
            [ nc, i ] = sort(abs(c.x),'descend');
            imax = i(1);
            dmax = c.dx(imax);
            j    = 2;
            while ( j<=max(v) && abs(c.x(imax))==abs(c.x(j)) )
              if ( c.dx(j)>dmax )
                imax = i(j);
                dmax = c.dx(imax);
              end
              j = j + 1;
            end
            obj = Deriv( nc(1), sign(c.x(imax))*dmax );
          else                                                   % p = -infinity
            [ nc, i ] = sort(abs(c.x),'ascend');
            imin = i(1);
            dmin = c.dx(imin);
            j    = 2;
            while ( j<=max(v) && abs(c.x(imin))==abs(c.x(j)) )
              if ( c.dx(j)<dmin )
                imin = i(j);
                dmin = c.dx(imin);
              end
              j = j + 1;
            end
            obj = Deriv( nc(1), sign(c.x(imin))*dmin );
          end
          
        else % the general p-norm case
          if ( ~isa(p,'Deriv') )
            nx  = norm(c.x,p);
            s   = abs(c);
            ndx = nx^((1-p)/p)*( sum( Get_value(s).^(p-1) .* Get_deriv(s) ) );
            obj = Deriv( nx, ndx );
          else
            error('not implemented yet')
          end
        end
        
      else
        %-----------------------------------------------------------------------
        % matrix norms
        %-----------------------------------------------------------------------
        if ( ischar(class(p)) )
          if ( strcmp(p,'fro') )                                     % \| A \|_F
            % Frobenius norm case
            n   = norm(c.x,p);
            obj = Deriv( n, sum(sum(c.x.*c.dx))/n );
            
          elseif ( strcmp(p,'1') || p==1 )                           % \| A \|_1
            % 1 norm case ( p interpreted as a string )
%             acs = sum(abs(c.x),1);
%             [n,i] = max(acs);
%             obj = Deriv( n, sum( sign(c.x(:,i)).*c.dx(:,i) ) );
            acs = sum(abs(c.x),1); % absolute column sum
            [n,i] = max(acs);
            ndx   = sum( sign(c.x(:,i)).*c.dx(:,i) );
            % test for pathelogical case
            for r=i+1:v
              if ( acs(i)-acs(r) < 1e-15*acs(i) )
                tmp = sum( sign(c.x(:,r)).*c.dx(:,r) ); % overloaded abs for complex?
                if ( tmp>ndx )
                  ndx = tmp;
                  i   = r;
                end
              end
            end
            obj = Deriv( n, ndx );
            
           elseif ( strcmp(p,'2') || p==2 )                          % \| A \|_2
            s1 = norm(c.x,2);
            % calculate u1 and v1
            n = size(c.x,1);
            %y = rand(2*n,1);
            [Q,~] = qr( [ -s1*eye(n) c.x'; c.x -s1*eye(n) ] );
            y = Q(:,end);%R \ (Q'*y);
            %y(1:n) = y(1:n)/norm(1:n);
            %y(n+1:end) = y(n+1:end)/norm(y(n+1:end));
            %y = R \ (Q'*y);
            v1 = y(1:n)/norm(y(1:n));
            u1 = y(n+1:end)/norm(y(n+1:end));
            s1_dx = u1'*c.dx*v1;
            obj = Deriv( s1, s1_dx );
            
          elseif ( strcmp(p,'inf') || isinf(p) )                % \| A \|_\infty
            ars = sum(abs(c.x),2); % absolute row sum
            [n,i] = max(ars);
            ndx   = sum( sign(c.x(i,:)).*c.dx(i,:) );
            % test for pathelogical case
            for r=i+1:v
              if ( ars(i)-ars(r) < 1e-15*ars(i) )
                tmp = sum( sign(c.x(r,:)).*c.dx(r,:) ); % overloaded abs for complex?
                if ( tmp>ndx )
                  ndx = tmp;
                  i   = r;
                end
              end
            end
            obj = Deriv( n, ndx );
            
          else  
            error('The only matrix norms available are 1, 2, inf, and ''fro''.')
            
          end

        else
          if ( p==1 )                                                % \| A \|_1
            acs = sum(abs(c.x),1); % absolute column sum
            [n,i] = max(acs);
            ndx   = sum( sign(c.x(:,i)).*c.dx(:,i) );
            % test for pathelogical case
            for r=i+1:v
              if ( rs(i)-rs(r) < 1e-15*rs(i) )
                tmp = sum( sign(c.x(:,r)).*c.dx(:,r) ); % overloaded abs for complex?
                if ( tmp>ndx )
                  ndx = tmp;
                  i   = r;
                end
              end
            end
            obj = Deriv( n, ndx );
            
          elseif ( p==2 )                                            % \| A \|_2
            s1 = norm(c.x,2);
            % calculate u1 and v1
            n = size(c.x,1);
            y = rand(2*n,1);
            y = [ -s1*eye(n) c.x'; c.x -s1*eye(n) ]\y;
            y(1:n) = y(1:n)/norm(1:n);
            y(n+1:end) = y(n+1:end)/norm(y(n+1:end));
            y = [ -s1*eye(n) c.x'; c.x -s1*eye(n) ]\y;
            v1 = y(1:n)/norm(1:n);
            u1 = y(n+1:end)/norm(y(n+1:end));
            s1_dx = u1'*c.dx*v1;
            obj = Deriv( s1, s1_dx );
            
          elseif ( isinf(p) )                                   % \| A \|_\infty
            ars = sum(abs(c.x),2); % absolute row sum
            [n,i] = max(ars);
            ndx   = sum( sign(c.x(i,:)).*c.dx(i,:) );
            % test for pathelogical case
            for r=i+1:size(c.x,1)
              if ( rs(i)-rs(r) < 1e-15*rs(i) )
                tmp = sum( sign(c.x(r,:)).*c.dx(r,:) ); % overloaded abs for complex?
                if ( tmp>ndx )
                  ndx = tmp;
                  i   = r;
                end
              end
            end
            obj = Deriv( n, ndx );
            
          else
            error('The only matrix norms available are 1, 2, inf, and ''fro''.')
            
          end
        end
      end
    end
    
    % normest
    
    % null
    
    % orth
    
    % rank
    
    % rcond
    
    % rref
    
    % subspace
    
    function obj = trace(c)
      obj = Deriv( trace(c.x), trace(c.dx) );
    end
    %%
    
    %---------------------------------------------------------------------------
    % Linear Equations p.1-39
    %% -------------------------------------------------------------------------
    % chol (below under Factorization)
    
    % cholinc (below under Factorization)
    
    % cond, condest (above)
    
    % funm
    
    % ichol (below under Factorization)
    
    % ilu (below under Factorization)
    
    function obj = inv(A)
      Ai  = inv(A.x);
      obj = Deriv( Ai, -Ai*A.dx*Ai );
    end

 
    % ldl (below under Factorization)
    
    % linsolve
    
    % lscov
    
    % lsqnonneg
    
    % lu
    
    % luinc
    
    function obj = pinv(A,tol)
      [U,S,V] = svd(A);
      clear A
      
      if ( nargin==1 )
        tol = max(size(S.x)) * S.x(1,1) * eps;
      end
      
      n = min(size(S.x));
      for i=1:min(size(S.x))
        if ( S.x(i,i) < tol )
          n = i-1;
          break
        end
      end
      
      Px  = zeros(size(S'));
      Pdx = Px;
      
      for i=1:n
        v   = ( U.x(:,i)' / S.x(i,i) );
        Px  = Px + V.x(:,i)*v ;
        Pdx = Pdx + V.dx(:,i)*v + V.x(:,i)*(U.dx(:,i)'/S.x(i,i)) ...
                  - V.x(:,i)*( (S.dx(i,i)/S.x(i,i)^2) * U.x(:,i)' );
      end
      clear U S V 
      
      obj = Deriv( Px, Pdx );
    end
    
    % qr (below under Factorization)
 
    % rcond (above)
    %%
    
    %---------------------------------------------------------------------------
    % Eigenvalues and Singular Values p.1-40
    %% -------------------------------------------------------------------------
    % balance
    
    % cdf2rdf
    
    % condeig (above)
    
    function [V,D] = eig(A)
      n = size(A.x,1);
      
      % Case: A.x is Hermitian
      if ( norm(A.x-A.x',inf)<1e-14 )
        [Vx,Dx] = eig(A.x);
        Ddx = zeros(n,n);
        for i=1:n
          Ddx(i,i) = Vx(:,i)'*A.dx*Vx(:,i);
        end
        if ( nargout==2 ) % eigenvectors are expected
          Vdx = zeros(n,n);
          
          % real and distinct case
          for i=1:n
            r = - ( A.dx-Ddx(i,i)*eye(n) )*Vx(:,i);
            r = Vx'*r;
            s = zeros(n,1);
            for j=1:i-1
              s(j) = r(j)/(Dx(j,j)-Dx(i,i));
            end
            for j=i+1:n
              s(j) = r(j)/(Dx(j,j)-Dx(i,i));
            end
            Vdx(:,i) = Vx*s;  % will leave out Vx(:,i) component
          end
        end
      else % non-Hermitian case
        
      end
      
      if ( nargout==1 )
        V = Deriv( diag(Dx), diag(Ddx) );
      else
        V = Deriv( Vx, Vdx );
        D = Deriv( Dx, Ddx );
      end
    end
    
    % eigs
    
    % gsvd
    
    function [Q,H] = hess(A)
      [m,n] = size(A.x);
      if (m~=n)
        error('Deriv:hess only applicable to a square matrix\n')
      end
      
      t   = zeros(m,1); t(1) = 1;
      V   = zeros(m,m);
      Vdx = zeros(m,m);
      H   = A.x;
      Hdx = A.dx;

      for k=1:m-2
        z      = H(k+1:m,k);
        zdx    = Hdx(k+1:m,k);
        norm_z = norm(z);
        v      = z + sign(z(1))*norm(z)*t(1:m-k);
        norm_v = norm(v);
        vdx    = zdx + sign(z(1))*zdx'*z/norm_z * t(1:m-k);
        vdx    = vdx/norm_v - v*(vdx'*v)/norm_v^3;
        v      = v/norm_v;

        temp   = v'*H(k+1:m,k:m);
        Hdx(k+1:m,k:m) = Hdx(k+1:m,k:m) - 2*vdx*temp - 2*v*(vdx'*H(k+1:m,k:m)) ...
                          - 2*v*(v'*Hdx(k+1:m,k:m));
        Hdx(k+2:m,k  ) = 0;

        H(k+1:m,k:m) = H(k+1:m,k:m) - 2*v*temp;
        H(k+2:m,k  ) = 0;
       
        temp           = H(1:m,k+1:m)*v;
        Hdx(1:m,k+1:m) = Hdx(1:m,k+1:m)                           ...
                       - 2*(Hdx(1:m,k+1:m)*v+H(1:m,k+1:m)*vdx)*v' ...
                       - 2*temp*vdx'; 
        H(1:m,k+1:m) = H(1:m,k+1:m) - 2*temp*v';
        
        
        V(k+1:m,k)   = v;
        Vdx(k+1:m,k) = vdx;
      end
      
      if ( nargout==2 )
        Q = eye(m);
        Qdx = zeros(m,m);
        for k=m-2:-1:1
          v   = V(k+1:m,k);
          vdx = Vdx(k+1:m,k);
          temp = v'*Q(k+1:m,:);
          Q(k+1:m,:) = Q(k+1:m,:) - 2*v*temp;
          Qdx(k+1:m,:) = Qdx(k+1:m,:) - 2*vdx*temp - 2*v*(vdx'*Q(k+1:m,:)+v'*Qdx(k+1:m,:));
        end
      
        H = Deriv( H, Hdx );
        Q = Deriv( Q, Qdx );
      else
        Q = Deriv( H, Hdx );
      end
    end
    
    % ordeig
    
    % ordqz
    
    % ordschur
    
    % poly
    
    % polyeig (below under Polynomials)
    
    % rsf2csf
    
    % schur
    
    % sqrtm (below in Matrix Logarithms and Exponentials)
    
    % ss2tf
    
    function [U,S,V] = svd(A,options)
      % SVD FACTORIZATION
      if ( nargin>1 )
        [Ux,Sx,Vx] = svd(A.x,options);
      else
        [Ux,Sx,Vx] = svd(A.x);
      end
      
      [sr,sc] = size(Sx);
      
      % Udx = zeros(size(Ux));
      Vdx = zeros(size(Vx));
      
      n = min(sr,sc);
      d = zeros(1,n);
      for i=1:n
        d(i) = Ux(:,i)'*A.dx*Vx(:,i);
      end
      Sdx = spdiags(d',0,sr,sc);
      
      % with the SVD of A.x, we could simplify the computation below...
      for i=1:n  % was sc
%        c        = (A.x'*A.x - Sx(i,i)^2*eye(sc)) \ ( 2*Sdx(i,i)*Sx(i,i)*Vx(:,i) - ( A.dx'*A.x+A.x'*A.dx )*Vx(:,i) );
        idx = [1:i-1, i+1:n];
        %eig( Vx(:,idx)*(Sx(idx,idx)'*Sx(idx,idx)-Sx(idx,idx)*speye(n-1))*Vx(:,idx)' ) 
        c        = (Vx(:,idx)*(Sx(idx,idx)'*Sx(idx,idx)-Sx(idx,idx)*speye(n-1))*Vx(:,idx)') \ ( 2*Sdx(i,i)*Sx(i,i)*Vx(:,i) - ( A.dx'*A.x+A.x'*A.dx )*Vx(:,i) );
        Vdx(:,i) = c - (c'*Vx(:,i))*Vx(:,i);
      end
      
      Udx = ( A.dx-Ux*Sdx*Vx'-Ux*Sx*Vdx' ) * Vx / Sx;
      
      U = Deriv( Ux, Udx );        clear Ux Udx;
      S = Deriv( Sx, full(Sdx) );  clear Sx Sdx;
      V = Deriv( Vx, Vdx );
    end

    % SVDS
    %%
    
    %---------------------------------------------------------------------------
    % Matrix Logarithms and Exponentials p.1-41
    %% -------------------------------------------------------------------------
    % expm
    
    % logm
    
    % sqrtm
    %%
    
    %---------------------------------------------------------------------------
    % Factorization p.1-41
    %% -------------------------------------------------------------------------
    function obj = chol(A)
      n  = size(A.x,1);
      Rx = chol(A.x);  
      Rdx = zeros(n,n);
      
      is_sym = norm(A.dx-A.dx',inf)<1e-15*norm(A.dx,inf);
      
      if ( is_sym )
        Rdx(1,1) = A.dx(1,1)/(2*Rx(1,1));
        Rdx(1,:) = ( ( A.dx(:,1)-Rdx(1,1)*Rx(1,:)' )/Rx(1,1) )';
        for i=2:n
          Rdx(i,i) = ( A.dx(i,i) - 2*Rdx(1:i-1,i)'*Rx(1:i-1,i) )/(2*Rx(i,i));
          Rdx(i,i:n) = ( ( A.dx(i:n,i) - Rx(1:i,i:n)'*Rdx(1:i,i) - Rdx(1:i-1,i:n)'*Rx(1:i-1,i) )/Rx(i,i) )';
        end
        
      else % no upper triangular solution is possible
        warning('Deriv called chol, but the derivative of the matrix is non-Hermitian')
        Rdx = lyap(Rx',-(A.dx+A.dx')/2); % a solution that isn't upper triangular
        %norm(Rx'*Rdx + Rdx'*Rx - A.dx)         
          
      end
          
      obj = Deriv( Rx, Rdx );
    end
    
    % cholinc
    
    % cholupdate
    
    % gsvd (above under Eigenvalues and Singular Value)
    
    % ichol
    
    % ilu
    
    % ldl
    
    % lu
    
    % luinc
    
    % planerot
    
    function [Q,R] = qr(A,options)
      % QR FACTORIZATION
      if ( nargin>1 )
        [Qx,Rx] = qr(A.x,options);
      else
        [Qx,Rx] = qr(A.x);
      end
      rc = size(Rx,2);
      
      Qdx = zeros(size(Qx));
      Rdx = zeros(size(Rx));  % same nonzero pattern
      
      % %  differentiating the modified Gram-Schmidt algorithm
      % for i=1:rc
      %   Rx(i,i) = norm(A.x(:,i));
      %   Rdx(i,i) = ( A.x(:,i)'*A.dx(:,i) ) / Rx(i,i);
      %   
      %   Qx(:,i) = A.x(:,i)/Rx(i,i);
      %   Qdx(:,i) = ( A.dx(:,i)*Rx(i,i) - A.x(:,i)*Rdx(i,i) ) / Rx(i,i)^2;
      %   
      %   for j=i+1:rc
      %     Rx(i,j) = Qx(:,i)'*A.x(:,j);
      %     Rdx(i,j) = Qdx(:,i)'*A.x(:,j) + Qx(:,i)'*A.dx(:,j);
      %     
      %     A.x(:,j) = A.x(:,j) - Rx(i,j)*Qx(:,i);
      %     A.dx(:,j) = A.dx(:,j) - Rdx(i,j)*Qx(:,i) - Rx(i,j)*Qdx(:,i);
      %   end
      % end
              
      Rdx(1,1) = Qx(:,1)'*A.dx(:,1);
      Qdx(:,1) = ( A.dx(:,1) - Qx(:,1)*Rdx(1,1) ) / Rx(1,1);
      
      for j=2:rc
        tmp        = Qx(:,1:j)'*Qdx(:,1:j);  % could just add the new portion of the matrix...
        tmp(:,j)   =-tmp(j,:);               % since Qdx(:,j) isn't known, use skew symmetry of Q.x' * Q.dx
        Rdx(1:j,j) = Qx(:,1:j)'*A.dx(:,j) - tmp*Rx(1:j,j);
        Qdx(:  ,j) = ( A.dx(:,j) - Qx(:,1:j)*Rdx(1:j,j) - Qdx(:,1:j-1)*Rx(1:j-1,j) )/Rx(j,j);
      end
      
      Q = Deriv( Qx, Qdx );
      R = Deriv( Rx, Rdx );
    end
    
    % qrdelete
    
    % qrinsert
    
    % qrupdate
    
    % qz
    
    % rsf2csf
    
    % svd (above under Eigenvalues and Singular Values)
    %%   
    
    %---------------------------------------------------------------------------
    % Elementary Math
    %---------------------------------------------------------------------------
    % Trigonometric p.1-42
    %% -------------------------------------------------------------------------
    function obj = acos(c)
      obj = Deriv( acos(c.x), -c.dx./sqrt(1-c.x.^2) );
    end
    
    % acosd
    
    function obj = acosh(c)
      obj = Deriv( acosh(c.x), c.dx./sqrt(c.x.^2-1) );
    end
    
    function obj = acot(c)
      obj = Deriv( acot(c.x), -c.dx./(c.x.^2+1) );
    end
    
    % acotd
    
    function obj = acoth(c)
      obj = Deriv( acoth(c.x), -c.dx.*csch(c.x).^2 );
    end
    
    function obj = acsc(c)
      obj = Deriv( acsc(c.x), -c.dx./( c.x.*sqrt(c.x.^2-1) ) );
    end
    
    % acscd
    
    function obj = acsch(c)
      obj = Deriv( acsch(c.x), -c.dx./( abs(c.x).*sqrt(c.x.^2+1) ) );
    end
    
    function obj = asec(c)
      obj = Deriv( asec(c.x), c.dx./( c.x.*sqrt(c.x.^2-1) ) );  
    end
    
    % asecd
    
    function obj = asech(c)
      obj = Deriv( asech(c.x), -c.dx./( c.x.*sqrt(1-c.x.^2) ) );
    end
    
    function obj = asin(c)
      obj = Deriv( asin(c.x), c.dx./sqrt(1-c.x.^2) );
    end
    
    % asind
    
    function obj = asinh(c)
      obj = Deriv( asinh(c.x), c.dx./sqrt(c.x.^2+1) );
    end
    
    function obj = atan(c)
      obj = Deriv( atan(c.x), c.dx./(c.x.^2+1) );  
    end
    
    % atan2
    
    % atan2d
    
    % atand
    
    function obj = atanh(c)
      obj = Deriv( atanh(c.x), c.dx./(1-c.x.^2) );
    end
    
    function obj = cos(c)
      obj = Deriv( cos(c.x), -sin(c.x).*c.dx );
    end
    
    function obj = cosd(c)
      obj = Deriv( cosd(c.x), -pi*sind(c.x).*c.dx/180 );
    end
    
    function obj = cosh(c)
      obj = Deriv( cosh(c.x), sinh(c.x).*c.dx );
    end
    
    function obj = cot(c)
      obj = Deriv( cot(c.x), -c.dx.*csc(c.x).^2 );
    end
    
    % cotd
    
    function obj = coth(c)
      obj = Deriv( coth(c.x), -c.dx.*csch(c.x).^2 );
    end
    
    function obj = csc(c)
      v   = csc(c.x);
      obj = Deriv( v, -c.dx.*v.*cot(c.x) );
    end
    
    % cscd
    
    % csch
    
    % hypot
    
    function obj = sec(c)
      v   = sec(c.x);
      obj = Deriv( v, v.*tan(c.x).*c.dx );
    end
    
    function obj = secd(c)
      v   = secd(c.x);
      obj = Deriv( v, -pi*tand(c.x).*v.*c.dx/180 );
    end
    
    function obj = sech(c)
      v   = sech(c.x);
      obj = Deriv( v, -tanh(c.x).*v.*c.dx );
    end
    
    function obj = sin(c)
      obj = Deriv( sin(c.x), cos(c.x).*c.dx );
    end
    
    function obj = sind(c)
      obj = Deriv( sind(c.x), pi*cosd(c.x).*c.dx/180 );
    end
    
    function obj = sinh(c)
      obj = Deriv( sinh(c.x), cosh(c.x).*c.dx );
    end
    
    function obj = tan(c)
      obj = Deriv( tan(c.x), (sec(c.x).^2).*c.dx );
    end
    
    function obj = tanh(c)
      obj = Deriv( tanh(c.x), (sech(c.x).^2).*c.dx );
    end
 
    %%
    
    %---------------------------------------------------------------------------
    % Exponential p.1-44
    %% -------------------------------------------------------------------------
    function obj = exp(c)
      obj = Deriv( exp(c.x), exp(c.x).*c.dx );
    end

    function obj = expm1(c)
      obj = Deriv( expm1(c.x), exp(c.x).*c.dx );
    end
    
    function obj = log(c)
      obj = Deriv( log(c.x), c.dx./c.x );
    end
    
    function obj = log10(c)
      obj = Deriv( log10(c.x), log10(exp(1))*c.dx./c.x );
    end
    
    % log1p
    
    function obj = log2(c)
      obj = Deriv( log2(c.x), log2(exp(1))*c.dx./c.x );
    end
    
    % nextpow2 (not diff)
    
    % pow2
    
    function obj = power(c,d)
      if ( ~isa(d,'Deriv') )
        obj = Deriv( c.x.^d, d.*c.x.^(d-1).*c.dx );
      elseif ( ~isa(c,'Deriv') )
        obj = Deriv( c.^d.x, c.^d.x .*d.dx * log(c) );
      else
        obj = Deriv( c.x.^d.x, d.x.*c.x.^(d.x-1).*c.dx ... 
                              +c.x.^d.x .*d.dx * log(c.x) );
      end
    end
        
    % reallog
    
    % realpow
    
    % realsqrt
    
    function obj = sqrt(c)
      tmp      = sqrt(c.x);
      idx      = find(tmp~=0);
      der      = Inf*ones(size(c.x));
      der(idx) = c.dx(idx)./(2*tmp(idx));
      obj = Deriv( tmp, der );
    end
    

    %%
    
    %---------------------------------------------------------------------------
    % Complex p.1-45
    %% -------------------------------------------------------------------------
    function obj = abs(c)
      if ( isreal(c.x) )
        obj = Deriv( abs(c.x), c.dx.*sign(c.x) ); 
%         c.x
%         c.dx
%         sign(c.x)%(c.x>0) - c.dx .* (c.x<0) );
      else
        a  = real(c.x);  adx = real(c.dx);
        b  = imag(c.x);  bdx = imag(c.dx);
        modul = abs(c.x);
        
        if ( modul~=0 )
          modul_dx = (a*adx+b*bdx)/modul;
        else
          warning('derivative of abs is not defined at zero')
          modul_dx = 0;
        end
        
        obj = Deriv( modul, modul_dx );
      end
      
      % not differentiable at zero, produces a warning in that case
      if ( sum(sum(c.x==0)) )
        warning('derivative of abs is not defined at zero')
      end
    end

    % angle
    
    % complex
    
    % conj
    
    % cplxpair
    
    % imag
    
    % isreal
    
    % real
    
    function out = sign(c)
      out = sign(c.x);
    end 
    
    % unwrap
    %%
    
    %---------------------------------------------------------------------------
    % Rounding and Remainder p.1-45
    %% -------------------------------------------------------------------------
    function out = ceil(c)
      % not differentiable, demote answer to an int
      out = ceil(c.x);
      warning('attempted to differentiate the function ceil')
    end
    
    function out = fix(c)
      % not differentiable, demote answer to a double
      out = fix(c.x);
      warning('attempted to differentiate the function fix')
    end
    
    function out = floor(c)
      % not differentiable, demote answer to an int
      out = floor(c.x);
      warning('attempted to differentiate the function floor')
    end
    
    % idivide (above)
    
    function obj = mod(c,d)
      error('function mod not implemented')
    end
    
    function obj = rem(c,d)
      error('function rem not implemented') 
    end
    
    function obj = round(c)
      % not differentiable, demote answer to an int
      out = round(c.x);
      warning('attempted to differentiate the function round')
    end
    %%

    %---------------------------------------------------------------------------
    % Discrete Math p.1-46
    %% -------------------------------------------------------------------------
    % factor (not diff)
    
    % factorial
    
    % gcd
    
    % isprime
    
    % lcm
    
    % nchoosek
    
    % perms
    
    % primes
    
    % rat
    
    % rats
    %%
    
    %---------------------------------------------------------------------------
    % Polynomials p.1-46
    %% -------------------------------------------------------------------------
    % conv
    
    % deconv
    
    % poly
    
    % polyder
    
    % polyeig
    
    % polyfit
    
    % polyint
    
    % polyval
    
    % polyvalm
    
    % residue
    
    % roots
    %%
    
    %---------------------------------------------------------------------------
    % Interpolation and Computational Geometry
    %---------------------------------------------------------------------------
    % Interpolation p.1-47
    %% -------------------------------------------------------------------------
    % dsearchn
    
    % griddata
    
    % griddatan
    
    % griddedInterpolant
    
    function obj = interp1(x,y,xi,method,extrapval)

      if ( ~isa(y,'Deriv') )  % set the derivative as zero for now
        if ( nargin==3 )
          yix = interp1(x,y,xi);
        elseif ( nargin==4 )
          yix = interp1(x,y,xi,method);
        elseif ( nargin==5 )
          yix = interp1(x,y,xi,method,extrapval);
        end
        yidx = zeros(size(yix));
        obj = Deriv( yix, yidx );
        return
      end
        
      % process inputs
      if ( nargin<4 || isempty(method) ) % set the default if necessary
        method = 'linear';
      end
      if ( strcmp(method,'pp') )
        error('the piecewise polynomial form is not yet implemented in interp1')
      end
      if ( strcmp(method,'v5cubic') )
        error('the method v5cubic is not implemented')
      end
  
      if ( nargin<5 )
        if ( strcmp(method,'nearest') || strcmp(method,'linear') )
          extrapval = NaN;
        end
      end
      
      % note that we assume x and xi are increasing (as in interp1q) if
      % not, then we should implement a loop to sort the inputs
      %
           
      ni   = length(xi);
      yix  = zeros(size(xi));
      yidx = zeros(size(xi));
      
      % Implementation of nearest neighbor
      if ( strcmp(method,'nearest') )
        index = zeros(1,ni);
        for i=1:ni
          [~,index(i)] = min( abs(x-xi(i)) );
          yix (i) = y.x (index(i));
          yidx(i) = y.dx(index(i));
        end
        
        obj = Deriv( yix, yidx );
      
      % Implementation of linear interpolation
      elseif ( strcmp(method,'linear') )
        for i=1:ni
          index = find( (x-xi(i))>0, 1 );
          if ( ~isempty(index) && index>1 )
            d       = ( x(index)-x(index-1) );
            a       = ( xi(i)-x(index-1) ) / d;
            b       = ( x(index)-xi(i)   ) / d;
            yix (i) = y.x(index-1)*b + y.x(index)*a;
            yidx(i) = y.dx(index-1)*b + y.dx(index)*a;
          elseif ( xi(i)==x(end) )
            yix (i) = y.x(end);
            yidx(i) = y.dx(end);
          else
            yix (i) = extrapval;
            yidx(i) = NaN;
          end
        end
        
        obj = Deriv( yix, yidx );
        
      % Implementation of Cubic spline interpolation
      elseif ( strcmp(method,'spline') )
        n = length(x);  
        h = diff(x);
        h = reshape(h,n-1,1);
        A = sparse(n,n);
        A(2:n-1,1:n) = spdiags([ h(1:n-2)  2*(h(1:n-2)+h(2:n-1)) h(2:n-1) ],0:2,n-2,n);
        A(1,1) = 1; A(n,n) = 1;
        B = zeros(n,2);
        B(2:n-1,1) = 3*(y.x (3:n)-y.x (2:n-1))'./h(2:n-1) - 3*(y.x (2:n-1)-y.x (1:n-2))'./h(1:n-2);
        B(2:n-1,2) = 3*(y.dx(3:n)-y.dx(2:n-1))'./h(2:n-1) - 3*(y.dx(2:n-1)-y.dx(1:n-2))'./h(1:n-2);
        c = A\B;
        
        for i=1:ni
          index = find( (x-xi(i))>0, 1 );
          if ( ~isempty(index) && index>1 )
            a    = y.x(index-1);
            h    = x(index)-x(index-1);
            b    = (y.x(index)-a)/h - h*(2*c(index-1,1)+c(index,1))/3;
            d    = (c(index,1)-c(index-1,1))/(3*h);
            ximx = ( xi(i) - x(index-1) );
            yix (i) = a + b*ximx + c(index-1,1)*ximx^2 + d*ximx^3;
            adx  = y.dx(index-1);
            bdx  = (y.dx(index)-adx)/h - h*(2*c(index-1,2)+c(index,2))/3;
            ddx  = (c(index,2)-c(index-1,2))/(3*h);
            yidx(i) = adx + bdx*ximx + c(index-1,2)*ximx^2 + ddx*ximx^3;
          elseif ( xi(i)==x(end) )
            yix (i) = y.x(end);
            yidx(i) = y.dx(end);
          else
            yix (i) = extrapval;
            yidx(i) = NaN;
          end
        end
        
        obj = Deriv( yix, yidx );
      % Implementation of Piecewise cubic Hermite interpolation
      elseif ( strcmp(method,'pchip') || strcmp(method,'cubic') )
        
        
      % Others not implemented
      else
        error('method %s not overloaded in interp1')
        
      end
    end
    
    % interp2
    
    % interp3
    
    % interpft
    
    % interpn
    
    % meshgrid (below under Domain Generation)
    
    % mkpp
    
    % ndgrid
    
    % padecoef
    
    % pchip
    
    % ppval
    
    % spline
    
    % TriScatteredInterp
    
    % tsearchn
    
    % unmkpp
    %%
    
    %---------------------------------------------------------------------------
    % Delaunay Triangulation p.1-48
    %% -------------------------------------------------------------------------
    % delaunay
    
    % delaunayn
    
    % size (TriRep)
    
    % trimesh
    
    % triplot
    
    % trisurf
    %%
    
    %---------------------------------------------------------------------------
    % Convex Hull p.1-48
    %% -------------------------------------------------------------------------
    % convexHull (DelaunayTri)
    
    % convhull
    
    % convhulln
    
    % patch (below under Plotting)
    
    % trisurf (above)
    
    %---------------------------------------------------------------------------
    % Voronoi Diagrams p.1-49
    %% -------------------------------------------------------------------------
    % patch (below under Plotting)
    
    % voronoi
    
    % voronoin
    
    %---------------------------------------------------------------------------
    % Domain Generation p.1-49
    %% -------------------------------------------------------------------------
    function obj = meshgrid(x,y,z)
      error('function meshgrid not implemented')
    end
    
    function obj = ndgrid(x1,x2,x3,x4,x5)
      error('function ndgrid not implemented')
    end
    
    %---------------------------------------------------------------------------
    % Cartesian Coordinate System Conversion p.1-49
    %% -------------------------------------------------------------------------
    % cart2pol
    
    % cart2sph
    
    % pol2cart
    
    % sph2cart
    %%
    
    %---------------------------------------------------------------------------
    % Nonlinear Numerical Methods
    %---------------------------------------------------------------------------
    % Ordinary Differential Equations p.1-50
    %% -------------------------------------------------------------------------
    %  Since adaptive time stepping algorithms are not generally differentiable
    %  we provide two approaches to overloading the Matlab ode suite.  
    %
    %  The first produces a result that is as consistent as possible by first 
    %  computing the solution to the original problem, then solving the 
    %  sensitivity equations using the time steps provided in the original solve.
    %  A limitation of this approach is that problems where the sensitivity
    %  equations are more stiff are not integrated as accurately.
    %
    %  The second uses the Matlab ode suite to integrate the coupled system
    %  of forward and sensitivity equations.  This is not consistent, but yields
    %  more accurate sensitivity solutions.  This option must be explicitly
    %  requested e.g., by calling Deriv_ode23 (instead of ode23).
    %
    %  Note that our strategy is to ignore the dependence of t on the 
    % parameter.  thus, s represents the partial derivative with respect
    % to parameter, by keeping t independent of the parameter, the returned
    % s would be consistent w/ the discrete derivative and maintain 
    % consistency of derived quantities.

    % decic
    
    % deval
    
    % ode113
    
    % ode15i
    
    % ode15s
    
    function [t,y,te,ye,ie] = ode23(odefun,tspan,y0,options)
      if ( nargin==3 )
        options = odeset();
      end
      
      [t,yx] = ode23(odefun,tspan,y0.x,options);
      
      % now integrate the sensitivity equation using the same time steps
      % following the 2nd order method outlined in bogacki1989a32.pdf
      [nsteps,ndim] = size(yx);
      s = zeros(nsteps,ndim);
      s(1,:) = y0.dx;
     
      for j=2:nsteps
        y0 = Deriv(yx(j-1,:),s(j-1,:));
        h  = t(j)-t(j-1);
        k1 = feval(odefun, t(j-1), y0 );
        tp = t(j-1)+h/2;
        yp = y0 + h*k1/2;
        k2 = feval(odefun, tp, yp);
        tp = t(j-1)+h*3/4;
        yp = y0 + h*(k2*3/4);
        k3 = feval(odefun, tp, yp);
        s(j,:) = s(j-1,:) + h*( (2/9)*k1.dx + (3/9)*k2.dx + (4/9)*k3.dx );
      end
      
      y = Deriv( yx, s );
      
    end
    
    % ode23s
    
    % ode23t
    
    % ode23tb
    
    % ode45
    function [t,y,te,ye,ie] = ode45(odefun,tspan,y0,options)
      if ( nargin==3 )
        options = odeset();
      end
      
      [t,yx] = ode45(odefun,tspan,y0.x,options);
      
      % now integrate the sensitivity equation using the same time steps
      % following the 4th order method of Dormand-Prince
      [nsteps,ndim] = size(yx);
      s = zeros(nsteps,ndim);
      s(1,:) = y0.dx;
     
      for j=2:nsteps
        t0 = t(j-1);
        y0 = Deriv(yx(j-1,:),s(j-1,:));
        h  = t(j)-t0;
        k1 = feval(odefun, t0, y0 );
        
        tp = t0 + h/5;
        yp = y0 + h*k1/5;
        k2 = feval(odefun, tp, yp);
        
        tp = t0 + h*3/10;
        yp = y0 + h*(k1*3/40+k2*9/40);
        k3 = feval(odefun, tp, yp);
        
        tp = t0 + h*4/5;
        yp = y0 + h*(k1*44/45-k2*56/15+k3*32/9);
        k4 = feval(odefun, tp, yp);
        
        tp = t0 + h*8/9;
        yp = y0 + h*(k1*19372/6561-k2*25360/2187+k3*64448/6561-k4*212/729);
        k5 = feval(odefun, tp, yp);
        
        tp = t(j);
        yp = y0 + h*(k1*9017/3168-k2*355/33+k3*46732/5247+k4*49/176-k5*5103/18656);
        k6 = feval(odefun, tp, yp);
        
        tp = t(j);
        yp = y0 + h*(k1*35/384+k3*500/1113+k4*125/192-k4*2187/6784+k5*11/84);
        k7 = feval(odefun, tp, yp);
        
        s(j,:) = s(j-1,:) + h*( (5179/57600)*k1.dx + (7571/16695)*k3.dx + (393/640)*k4.dx - (92097/339200)*k5.dx + (187/2100)*k6.dx + (1/40)*k7.dx);
      end
      
      y = Deriv( yx, s );
       
    end
    
    
    % odeget
    
    % odeset
    
    % odextend
    %%
    
    %---------------------------------------------------------------------------
    % Delay Differential Equations p.1-51
    %% -------------------------------------------------------------------------
    % dde23
    
    % ddeget
    
    % ddensd
    
    % ddesd
    
    % ddeset
    
    % deval
    
    %%
    
    %---------------------------------------------------------------------------
    % Boundary Value Problems p.1-52
    %% -------------------------------------------------------------------------
    % bvp4c
    
    % bvp5c
    
    % bvpget
    
    % bvpinit
    
    % bvpset
    
    % bvpxtend
    
    % deval
    
    %%
    
    %---------------------------------------------------------------------------
    % Partial Differential Equations p.1-52
    %% -------------------------------------------------------------------------
    % pdepe
    
    % pdeval
    
    %%
    
    %---------------------------------------------------------------------------
    % Optimization p.1-53
    %% -------------------------------------------------------------------------
    % fminbnd
    
    % fminsearch
    
    % fzero
    
    % lsqnonneg
    
    % optimget
    
    % optimset
    
    %%
    
    %---------------------------------------------------------------------------
    % Numerical Integration p.1-53
    %% -------------------------------------------------------------------------
    % dblquad
    
    % integral
    
    % integral2
    
    % integral3
    
    % quad
    
    % quad2d
    
    % quadgk
    
    % quadl
    
    % quadv
    
    % triplequad
    %%
    
    
    %---------------------------------------------------------------------------
    % Special Functions p.1-54
    %% -------------------------------------------------------------------------
    % airy, besselh, besseli, besselj, besselk, bessely, beta, betainc,
    % betaincinv, betaln, ellipj, ellipke, erf, erfc, erfcinv, erfcx,
    % erfinv, expint, gamma, gammainc, gammaincinv, gammaln, legendre
    
    %%
    
    %---------------------------------------------------------------------------
    % Sparse Matrices
    %---------------------------------------------------------------------------
    % Elementary Sparse Matrices p.1-55
    %% -------------------------------------------------------------------------
    % spdiags, speye, sprand, sprandn, sprandsym
    
    
    %%
    %---------------------------------------------------------------------------
    % Full to Sparse Conversion p.1-56
    %% -------------------------------------------------------------------------
    % find, spconvert
    function obj = full(c)
      obj = Deriv( full(c.x), full(c.dx) );
    end
    
    function obj = sparse(varargin)
      bypass = Deriv( 1, 1 );
      [X,DX] = process_varargin(varargin,bypass);
      n_in = length(X);
      
      if ( n_in==1 )
        obj = Deriv( sparse(X{1}), sparse(DX{1}) );
      elseif ( n_in==2 && isscalar(X{1}) && isscalar(X{2}) ) % accidental promotion
        obj = sparse( X{1}, X{2} );
      elseif ( n_in==3 )
        obj = Deriv( sparse(X{1},X{2},X{3}), sparse(X{1},X{2},DX{3}) );
      elseif ( n_in==5 )
        obj = Deriv( sparse(X{1},X{2}, X{3},X{4},X{5}), ...
                     sparse(X{1},X{2},DX{3},X{4},X{5}) );
      elseif ( n_in==6 )
         obj = Deriv( sparse(X{1},X{2}, X{3},X{4},X{5},X{6}), ...
                      sparse(X{1},X{2},DX{3},X{4},X{5},X{6}) );
      else
        warning('Deriv: sparse called with unusual input arguments')
        obj = [];
      end

    end
    
    %%
    %---------------------------------------------------------------------------
    % Sparse Matrix Manipulation p.1-56
    %% -------------------------------------------------------------------------
    % issparse (above), 
    % nnz, nonzeros, nzmax, spalloc, spfun, spones, spparms, spy
    
    %%
    %---------------------------------------------------------------------------
    % Reordering Algorithms p.1-57
    %% -------------------------------------------------------------------------
    % amd, colamd, colperm, dmperm, ldl, randperm, symamd, symrcm
    %%
    %---------------------------------------------------------------------------
    % Linear Algebra p.1-57
    %% -------------------------------------------------------------------------
    % cholinc, condest, eigs, ichol, ilu, luinc, normest, spaugment, sprand, svds
    
    %%
    %---------------------------------------------------------------------------
    % Linear Equations (Iterative Methods) p.1-58
    %% -------------------------------------------------------------------------
    % bicg, bicgstab, bicgstabl, cgs, gmres, lsqr, minres, pcg, qmr, symmlq, tfqmr
    
    %%
    %---------------------------------------------------------------------------
    % Tree Operations p.1-58
    %% -------------------------------------------------------------------------
    % etree, etreeplot, gplot, symbfact, treelayout, treeplot, unmesh
    
    %---------------------------------------------------------------------------
    % Data Analysis
    %---------------------------------------------------------------------------
    % Basic Operations p.1-60
    %% -------------------------------------------------------------------------
    % brush, cumprod, cumsum, linkdata, prod, sort, sortrows, sum, 
    
    %%
    %---------------------------------------------------------------------------
    % Descriptive Statistics p.1-60
    %% -------------------------------------------------------------------------
    % corrcoef, cov, max, mean, median, min, mode, std, var
    %%
    %---------------------------------------------------------------------------
    % Filtering and Convolution p.1-61
    %% -------------------------------------------------------------------------
    % conv, conv2, convn, deconv, filter, filter2
    %%
    %---------------------------------------------------------------------------
    % Interpolation and Regression p.1-61
    %% -------------------------------------------------------------------------
    % interp1, interp2, interp3, interpn, polyfit, polyval
    %%
    %---------------------------------------------------------------------------
    % Fourier Transforms p.1-62
    %% -------------------------------------------------------------------------
    % abs, angle, cplxpair, fft, fft2, fftn, fftshift, fftw, ifft, ifft2, ifftn, 
    % ifftshift, nextpow2, unwrap
    
    %%
    %---------------------------------------------------------------------------
    % Derivatives and Integrals p.1-62
    %% -------------------------------------------------------------------------
    % cumtrapz, del2, diff, gradient, polyder, polyint, trapz
    
    %%
    %---------------------------------------------------------------------------
    % Time Series Objects p.1-63
    %% -------------------------------------------------------------------------
    % append, get, getdatasamples, getdatasamplesize, getqualitydesc, getsamples,
    % plot, set, timeseries, tsdata.event, tstool
    
    % addsample, ctranspose, delsample, detrend, filter, getabstime, getinterpmethod,
    % getsampleusingtime, idealfilter, resample, setabstime, setinterpmethod,
    % setuniformtime, synchronize, transpose
    
    % addevent, delevent, getsafteratevent, gettsafterevent, getsatevent,
    % gettsbeforeatevent, gettsbeforeevent, gettsbetweenevents
    
    % iqr...
    
    
    %%
    %---------------------------------------------------------------------------
    % Descriptive Statistics p.1-60
    %% -------------------------------------------------------------------------

    function obj = mtimes(c,d)   % c*d
      if ( ~isa(c,'Deriv') )
        obj = Deriv( c*d.x, c*d.dx );
      elseif ( ~isa(d,'Deriv') )
        obj = Deriv( c.x *d, c.dx*d );
      else
        obj = Deriv( c.x*d.x, c.x*d.dx + c.dx*d.x );
      end 
    end
    
    function obj = rdivide(c,d)   % c./d
      if ( ~isa(c,'Deriv') )
        obj = Deriv( c./d.x, -(c.*d.dx)./(d.x.^2) );
      elseif ( ~isa(d,'Deriv') )
        obj = Deriv( c.x./d, (c.dx.*d)./(d.^2) );
      else
        obj = Deriv( c.x./d.x, ( (c.dx.*d.x - c.x.*d.dx) ./ (d.x.^2) ) );
      end
    end
    
    function obj = ldivide(c,d)   % c.\d
      if ( ~isa(c,'Deriv') )
        obj = Deriv( c.\d.x, c.\d.dx );
      elseif ( ~isa(d,'Deriv') )
        obj = Deriv( c.x.\d, -(c.x.^2) .\ (c.dx.*d) );
      else
        obj = Deriv( c.x .\ d.x, (c.x.^2) .\ (-c.dx.*d.x + c.x.*d.dx) );
      end
    end
    
    function obj = mrdivide(c,d)  % solve x*c=d for x
      % need to add backward solve case...
      if ( ~isa(c,'Deriv') )
        obj = Deriv( c/d.x, -c*d.dx/d.x^2 );
      elseif ( ~isa(d,'Deriv') )
        obj = Deriv( c.x/d, c.dx*d/d^2 );
      else
        obj = Deriv( c.x/d.x, (c.dx*d.x - c.x*d.dx)/d.x^2 );
      end
    end
    
    function obj = mldivide(A,b)  % A\b     
      % add dense version exploiting factorizations
      if (~isa(A,'Deriv'))
        ar = size(A,1);
        [br,bc] = size(b.x);
        
        if ( ar ~= br )
          error('Incompatible system dimensions')
          return
        end
        
        m   = A \ [ b.x b.dx ];
        obj = Deriv( m(:,1:bc), m(:,bc+1:end) );
        
      elseif ( ~isa(b,'Deriv') )
        m   = A.x \ b;
        tmp = -A.x\(A.dx*m);
        obj = Deriv( m, tmp );
        
      else
        m   = A.x \ b.x;
        tmp = A.x\(b.dx - A.dx*m);
        obj = Deriv( m, tmp );
      end
    end
    
    %%    
     
    %---------------------------------------------------------------------------
    % Overload Basic Constants (replace eye with Deye and zeros with Dzeros)
    %---------------------------------------------------------------------------

    function out = double(x)
      out = Get_value(x);
    end
    
    % See Matlab: Language Fundamentals: Operators and Elementary
    % Operations: Arithmetic
    %---------------------------------------------------------------------------
    % Overload Arithmetic Operators
    %---------------------------------------------------------------------------
    

    function obj = diff(c,n,dim)
      if ( nargin==1 )
        obj = Deriv( diff(c.x), diff(c.dx) );
      elseif ( nargin==2 )
        obj = Deriv( diff(c.x,n), diff(c.dx,n) );
      else
        obj = Deriv( diff(c.x,n,dim), diff(c.dx,n,dim) );
      end
    end
    
    
    % See Matlab: Language Fundamentals: Operators and Elementary
    % Operations: Relational Operations
    %---------------------------------------------------------------------------
    % Overload Relational Operators
    %---------------------------------------------------------------------------
    function bool = eq(c,d)
      if ( ~isa(c,'Deriv') )
        bool = ( c == d.x );
      elseif ( ~isa(d,'Deriv') )
        bool = ( c.x == d );
      else
        bool = ( c.x == d.x );
      end
    end
    
    function bool = ge(c,d)
      if ( ~isa(c,'Deriv') )
        bool = ( c >= d.x );
      elseif ( ~isa(d,'Deriv') )
        bool = ( c.x >= d );
      else
        bool = ( c.x >= d.x );
      end
    end
    
    function bool = gt(c,d)
      if ( ~isa(c,'Deriv') )
        bool = ( c > d.x );
      elseif ( ~isa(d,'Deriv') )
        bool = ( c.x > d );
      else
        bool = ( c.x > d.x );
      end
    end
    
    function bool = le(c,d)
      if ( ~isa(c,'Deriv') )
        bool = ( c <= d.x );
      elseif ( ~isa(d,'Deriv') )
        bool = ( c.x <= d );
      else
        bool = ( c.x <= d.x );
      end
    end
    
    function bool = lt(c,d)
      if ( ~isa(c,'Deriv') )
        bool = ( c < d.x );
      elseif ( ~isa(d,'Deriv') )
        bool = ( c.x < d );
      else
        bool = ( c.x < d.x );
      end
    end
    
    function bool = ne(c,d)
      if ( ~isa(c,'Deriv') )
        bool = ( c ~= d.x );
      elseif ( ~isa(d,'Deriv') )
        bool = ( c.x ~= d );
      else
        bool = ( c.x ~= d.x );
      end
    end
    
    % See Matlab: Language Fundamentals: Operators and Elementary
    % Operations: Logical Operations
    %---------------------------------------------------------------------------
    % Overload Logical Operators: Elementwise and Short-circuit
    %---------------------------------------------------------------------------
    function bool = and(c,d)
      bool = and(c.x,d.x);
    end
    
    function bool = not(c)
      bool = ~c.x;
    end
    
    function bool = or(c,d)
      if ( ~isa(c,'Deriv') )
        bool = ( c | d.x );
      elseif ( ~isa(d,'Deriv') )
        bool = ( c.x | d );
      else
        bool = ( c.x | d.x );
      end
    end
    
    function bool = xor(c,d)
      if ( ~isa(c,'Deriv') )
        bool = xor(c,d.x);
      elseif ( ~isa(d,'Deriv') )
        bool = xor(c.x,d);
      else
        bool = xor( c.x, d.x );
      end
    end
    
    function bool = all(c)
      bool = all( c.x );
    end
    
    function bool = any(c)
      bool = any( c.x );
    end
    
    function [i,j,obj] = find(c,k,opt)
      error('function find not implented')
    end
    
    % leaving out false, logical, and true
    
    % See Matlab: Language Fundamentals: Operators and Elementary
    % Operations: Set Operations
    %---------------------------------------------------------------------------
    % Overload Set Operators
    %---------------------------------------------------------------------------
    % intersect, ismember, issorted, setdiff, setxor, union, unique
    
    
    % leaving out all bit-wise operations
    
    
    % See Matlab: Language Fundamentals: Matrices and Arrays
    %---------------------------------------------------------------------------
    % Overload Operators Associated with Array Creation and Concatenation
    %---------------------------------------------------------------------------
    % accumarray, blkdiag, diag, eye, linspace, logspace, meshgrid, ndgrid,
    % ones, zeros, cat, horzcat, vertcat

  
    
    % See Matlab: Language Fundamentals: Matrices and Arrays: Indexing
    %---------------------------------------------------------------------------
    % Overload Operators Associated with Array Indexing
    %---------------------------------------------------------------------------
 
    
    % See Matlab: Language Fundamentals: Matrices and Arrays: Array
    % Dimensions
    %---------------------------------------------------------------------------
    % Overload Operators Associated with Array Dimensions
    %---------------------------------------------------------------------------

    
    
 
        
    %---------------------------------------------------------------------------
    % Define derivatives of functions using product rules
    %---------------------------------------------------------------------------
    
    %---------------------------------------------------------------------------
    % Functions affecting basic Matlab behavior
    %---------------------------------------------------------------------------
    function obj = imag(c)  % c(x) = r(x) + i i(x), c' = r' + i i' 
      obj = Deriv( imag(c.x), imag(c.dx) );
    end
    
    function [m] = rank(c)
      m = rank(c.x);
    end
    
    function obj = real(c)
      obj = Deriv( real(c.x), real(c.dx) );
    end
    
    function out = sprintf(format,a1,a2,a3,a4)
      if ( nargin==2 )
        out = builtin('sprintf',format,a1.x);
      elseif ( nargin==3 )
        if ( ~isa(a1,'Deriv') )
          out = builtin('sprintf',format,a1,a2.x);
        elseif ( ~isa(a2,'Deriv') )
          out = builtin('sprintf',format,a1.x,a2);
        else
          out = builtin('sprintf',format,a1.x,a2.x);
        end
      elseif ( nargin==4 )
        if ( ~isa(a1,'Deriv') && ~isa(a2,'Deriv') )
          out = builtin('sprintf',format,a1,a2,a3.x);
        elseif ( ~isa(a1,'Deriv') && ~isa(a3,'Deriv') )
          out = builtin('sprintf',format,a1,a2.x,a3);
        elseif ( ~isa(a2,'Deriv') && ~isa(a3,'Deriv') )
          out = builtin('sprintf',format,a1.x,a2,a3);
        elseif ( ~isa(a1,'Deriv') )
          out = builtin('sprintf',format,a1,a2.x,a3.x);
        elseif ( ~isa(a2,'Deriv') )
          out = builtin('sprintf',format,a1.x,a2,a3.x);
        elseif ( ~isa(a3,'Deriv') )
          out = builtin('sprintf',format,a1.x,a2.x,a3);
        else
          out = builtin('sprintf',format,a1.x,a2.x,a3.x);
        end
      else % partially implemented
        out = builtin('sprintf',format,a1.x,a2.x,a3.x,a4.x);  
      end
    end
    
    function [] = spy(c)
      spy(c.x)
    end
    
    function [] = surf(x,y,z,c)
      if ( ~isa(x,'Deriv') )
        x = Deriv(x);
      end
      
      if ( ~isa(y,'Deriv') )
        y = Deriv(y);
      end
      
      if ( ~isa(z,'Deriv') )
        z = Deriv(z);
      end
      
      if (~isa(c,'Deriv') )
        c = Deriv(c);
      end
      
      surf(x.x,y.x,z.x,c.x)
    end
    
    function A = subsasgn(A,S,B)
      if ( ~isa(A,'Deriv') )  % promote A to a Deriv object
        A = Deriv( A, zeros(size(A)) );
        A.x  = subsasgn( A.x , S, B.x  );
        A.dx = subsasgn( A.dx, S, B.dx );
        
      elseif ( ~isa(B,'Deriv') )
        A.x  = subsasgn( A.x , S, B              );
        A.dx = subsasgn( A.dx, S, zeros(size(B)) );
        
      else
        %fprintf('in subsasgn')
        %size(A.x)
        %size(B.x)
        %S
        A.x  = subsasgn( A.x , S, B.x  );
        A.dx = subsasgn( A.dx, S, B.dx );
        
      end
    end
    
    function out = subsindex(c)
      %  Implemented for cases when indices are "accidently" promoted to Deriv
      out = c.x-1;
    end


    function obj = subsref(c,index)
%        fprintf('in subsref\n')
%        disp(c)
%        disp(index)
       switch index.type
         case '()'
           obj = Deriv( c.x(index.subs{:}), c.dx(index.subs{:}) );
         case '{}'
           obj = Deriv( c.x.CellProperty{index.subs{:}}, c.dx.CellProperty{index.subs{:}} );
         case '.'
           obj = builtin('subsref',c,index);
       end
           
    end
        
    
    
    %---------------------------------------------------------------------------
    % Boolean Functions
    %---------------------------------------------------------------------------

    
    %---------------------------------------------------------------------------
    % Differentiating Matrix Factorizations / Solvers
    %---------------------------------------------------------------------------
    % CARE
    function [X,L,G] = care(A,B,Q,R,S,E)
      n = size(A,2);
      m = size(B,2);
      if ( nargin<4 )
        Rinv = Deriv( eye(m), zeros(m,m) );
      else
        if ( isa(R,'Deriv') )
          Rinv = inv(R.x);
        else
          Rinv = inv(R);
          R    = Deriv( R, zeros(size(R)) );
        end
      end
      
      if ( nargin<5 )
        [Xx,Lx,Gx] = care(A.x,B.x,Q.x,R.x);
        
        [Xdx] = lyap( A.x'-Xx*B.x*Rinv*B.x' , ...
               A.dx'*Xx + Xx*A.dx - Xx*( B.dx*Rinv*B.x' + B.x*Rinv*B.dx' ...
               - B.x*Rinv*R.dx*Rinv*B.x' )*Xx + Q.dx );
        X = Deriv( Xx, Xdx );
        L = [];
        G = [];
      end
      
    
    end
    
    % EIG
    
    % EIGS
    
    % LQR
    function [K,X,E] = lqr(A,B,Q,R,N)
      if ( isa(A,'ss') )
        error(' Deriv class does not currently overload the ss class')
        return
      end
      
      if ( nargin==3 )
        [X,L,G] = care(A,B,Q);
        m = size(B,2);
        R = eye(m);
      elseif ( nargin==4 )
        [X,L,G] = care(A,B,Q,R);
      elseif ( nargin==5 )
        [X,L,G] = care(A,B,Q,R,N);
      else
        error(' Deriv class implementation of lqr expects 3,4, or 5 arguments')
      end
      
      if ( isa(R,'Deriv') )
        K = Deriv( (R.x\B.x')*X.x, R.x\(B.dx'*X.x + B.x'*X.dx - R.dx*(R.x\B.x')*X.x) );
      else
        K = Deriv( (R\B.x')*X.x, R\(B.dx'*X.x + B.x'*X.dx) );
      end
      
      E = [];
    end
      
    % LYAPUNOV SOLVER
    %  We are repeatedly using Matlab's Lyapunov solver rather than rewriting
    %  a solver that could reuse the factorization of A.
    function obj = lyap(A,B,C,D)
      if ( nargin==2 )  %  Lyapunov  AX + XA' + B = 0  case
        Xx  = lyap( A.x, B.x );
        Xdx = lyap( A.x, A.dx*Xx + Xx*A.dx' + B.dx );
        clear A B
 
      elseif ( nargin==3 )  %  Sylvester   AX + XB + C = 0  case
        Xx  = lyap( A.x, B.x, C.x );
        Xdx = lyap( A.x, B.x, A.dx*Xx + Xx*B.dx + C.dx );
        clear A B C
        
      elseif ( nargin==4 )  %  Generalized Lyapunov  AXD' + DXA' + B = 0  case
        Xx  = lyap( A.x, B.x, [], D.x );
        E   = A.dx*Xx*D.x' + A.x*Xx*D.dx' + D.dx*Xx*A.x' + D.x*Xx*A.dx' + B.dx;
        % don't know how "non self adjoint" E is
        Xdx = lyap( A.x, (E+E')/2, [], D.x );
        clear A B C D
        
      end
      
      obj = Deriv( Xx, Xdx );
    end
    
    
    
   
    
    %---------------------------------------------------------------------------
    % Linear System Solves
    %---------------------------------------------------------------------------
    % \
    % GMRES
    
    
    
      
    %---------------------------------------------------------------------------
    % Overloaded ODE Solvers
    %---------------------------------------------------------------------------
    % ODE23
    
    % ODE45
    
    
    %---------------------------------------------------------------------------
    % "Nondifferentiable Functions"
    %---------------------------------------------------------------------------
    % abs ----------------------------------------------------------------------
    

  end % methods
  
end
