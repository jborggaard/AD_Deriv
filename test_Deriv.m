function [ success ] = test_Deriv(  )
% Runs a series of finite difference tests to verify the Deriv Matlab class.

  step = 1e-9;   % finite difference step size
  tol  = 2.2e-6;   % relative tolerance that must be met to pass the test
  
  verbose = false;   % flag that determines the output level
  success = true;
  
  tcase.arithmetic  = 1;
  tcase.array_manip = 1;
  tcase.sparse      = 1;
  
  tcase.chol        = 1;  % matrix factorizations / equations
  tcase.eig         = 0;
  tcase.lqr         = 1;
  tcase.lyap        = 1;
  tcase.norm        = 1;
  tcase.hess        = 1;
  tcase.qr          = 1;
  tcase.svd         = 1;
  
  tcase.abs         = 1;  % nondifferentiable Matlab functions
  tcase.interp1     = 1;
  tcase.ode         = 0;
  
  %% ---------------------------------------------------------------------------
  %  Testing arithmetic operations
  %%----------------------------------------------------------------------------
  if (tcase.arithmetic)
    fprintf(' Testing arithmetic operations\n')
    fprintf('   ldivide:\n')
    n  = 3;
    A  = Deriv( rand(n,n), rand(n,n) );  
    B  = Deriv( rand(n,n), rand(n,n) );
    c  = Deriv( rand(n,1), rand(n,1) );
    Ap = Get_value(A) + step*Get_deriv(A);
    Bp = Get_value(B) + step*Get_deriv(B);
    cp = Get_value(c) + step*Get_deriv(c);
    C  = rand(n,n);
    
    D  = A.\B;  Dp = Ap.\Bp;
    rel_error1 = norm( (Dp-Get_value(D))/step - Get_deriv(D) )/norm( Get_deriv(D) );
    D  = A.\C;  Dp = Ap.\C;
    rel_error2 = norm( (Dp-Get_value(D))/step - Get_deriv(D) )/norm( Get_deriv(D) );
    D  = C.\A;  Dp = C.\Ap;
    rel_error3 = norm( (Dp-Get_value(D))/step - Get_deriv(D) )/norm( Get_deriv(D) );
    
    if ( verbose )
      fprintf('    Deriv  ldivide Deriv  test: the relative error is %g\n',rel_error1);
      fprintf('    Deriv  ldivide double test: the relative error is %g\n',rel_error2);
      fprintf('    double ldivide Deriv  test: the relative error is %g\n',rel_error3);
    end
    if ( rel_error1 > tol || rel_error2 > tol )
      fprintf('  ldivide test failed\n')
      fprintf('   Deriv  ldivide Deriv  test: relative error %g\n',rel_error1);
      fprintf('   Deriv  ldivide double test: relative error %g\n',rel_error2);
      fprintf('   double ldivide Deriv  test: relative error %g\n',rel_error3);
      success = false;
    end
    
    
    fprintf('   cumprod:\n')
    d = cumprod(c); dp = cumprod(cp);
    rel_error1 = norm( (dp-Get_value(d))/step - Get_deriv(d) )/norm( Get_deriv(d) );
        
    d = cumprod(A,1); dp = cumprod(Ap,1);
    rel_error2 = norm( (dp-Get_value(d))/step - Get_deriv(d) )/norm( Get_deriv(d) );
    d = cumprod(A,2); dp = cumprod(Ap,2);
    rel_error3 = norm( (dp-Get_value(d))/step - Get_deriv(d) )/norm( Get_deriv(d) );
    if ( verbose )
      fprintf('    vector test: the relative error is %g\n',rel_error1);
      fprintf('    matrix test: the relative error is %g\n',rel_error2);
      fprintf('    matrix test: the relative error is %g\n',rel_error3);
    end
    if ( rel_error1 > tol || rel_error2 > tol || rel_error3 > tol )
      fprintf('  cumprod test failed\n')
      fprintf('   vector test: relative error %g\n',rel_error1);
      fprintf('   matrix test: relative error %g\n',rel_error2);
      fprintf('   matrix test: relative error %g\n',rel_error3);
      success = false;
    end
    
    
    fprintf('   prod:\n')
    d = prod(c); dp = prod(cp);
    rel_error1 = norm( (dp-Get_value(d))/step - Get_deriv(d) )/norm( Get_deriv(d) );
    if ( verbose )
      fprintf('    vector test: the relative error is %g\n',rel_error1);
    end
    if ( rel_error1 > tol )
      fprintf('  prod test failed\n')
      fprintf('   vector test: relative error %g\n',rel_error1);
      success = false;
    end
    
    fprintf('   diff:\n')
    d = diff(c); dp = diff(cp);
    rel_error1 = norm( (dp-Get_value(d))/step - Get_deriv(d) )/norm( Get_deriv(d) );
    if ( verbose )
      fprintf('    standard test: the relative error is %g\n',rel_error1);
    end
    
    fprintf('\n')    
  end
  
  %% ---------------------------------------------------------------------------
  %  Testing Array Manipulations
  %%----------------------------------------------------------------------------
  if ( tcase.array_manip )
    fprintf(' Testing array manipulations\n');
    Ax = rand(2,1); Adx = rand(2,1); A = Deriv(Ax,Adx);
    Bx = rand(2,3); Bdx = rand(2,3); B = Deriv(Bx,Bdx);
    Cx = rand(2,2); Cdx = rand(2,2); C = Deriv(Cx,Cdx);
    
    
    fprintf('   cat:\n')
    M   = cat(2,A,B,C);
    Mx  = cat(2,Ax,Bx,Cx);
    Mdx = cat(2,Adx,Bdx,Cdx);
    error1 = norm( Get_value(M)-Mx );
    error2 = norm( Get_deriv(M)-Mdx );
    if ( verbose )
      fprintf('    cat test: the error is %g\n',error1);
      fprintf('    cat test: the error is %g\n',error2);
    end
    if ( error1 > tol || error2 > tol )
      fprintf('  cat test failed\n')
      fprintf('   first  test: error %g\n',error1);
      fprintf('   second test: error %g\n',error2);
      success = false;
    end
    
    fprintf('   horzcat:\n');
    M   = [A   B   C   C   C  ];
    Mx  = [Ax  Bx  Cx  Cx  Cx ];
    Mdx = [Adx Bdx Cdx Cdx Cdx];
    error1 = norm( Get_value(M)-Mx );
    error2 = norm( Get_deriv(M)-Mdx );
    if ( verbose )
      fprintf('    cat test: the error is %g\n',error1);
      fprintf('    cat test: the error is %g\n',error2);
    end
    if ( error1 > tol || error2 > tol )
      fprintf('  cat test failed\n')
      fprintf('   first  test: error %g\n',error1);
      fprintf('   second test: error %g\n',error2);
      success = false;
    end
    
    fprintf('   vertcat:\n');
    M   = [A'   ; B'  ; C  ; C  ; C  ];
    Mx  = [Ax'  ; Bx' ; Cx ; Cx ; Cx ];
    Mdx = [Adx' ; Bdx'; Cdx; Cdx; Cdx];
    error1 = norm( Get_value(M)-Mx );
    error2 = norm( Get_deriv(M)-Mdx );
    if ( verbose )
      fprintf('    cat test: the error is %g\n',error1);
      fprintf('    cat test: the error is %g\n',error2);
    end
    if ( error1 > tol || error2 > tol )
      fprintf('  cat test failed\n')
      fprintf('   first  test: error %g\n',error1);
      fprintf('   second test: error %g\n',error2);
      success = false;
    end
    
    fprintf('\n')    
  end
  
  %% ---------------------------------------------------------------------------
  %  Testing Matrix Functions
  %%----------------------------------------------------------------------------
  %  CHOL
  %-----------------------------------------------------------------------------
  if (tcase.chol || tcase.eig || tcase.lqr || tcase.lyap || tcase.hess || ...
      tcase.qr   || tcase.svd )
    fprintf(' Testing matrix factorizations\n');
  end
  
  
  if (tcase.chol)
    fprintf('   chol:\n')
    n = 50;
    X = randn(n,n);  Y = randn(n,n);
    
    % symmetric case (differentiable)
    fprintf('     symmetric derivative case\n')
    Ax = X*X';  Adx = Y+Y'; % subcase1: symmetric matrix derivative
    A  = Deriv( Ax, Adx );
    
    [R] = chol(A);
    [Rp] = chol(Ax + step*Adx);
    rel_error1s = norm( (Rp-Get_value(R))/step - Get_deriv(R) )/norm( Get_deriv(R) );

%     Adx = Y; % subcase2: nonsymmetric matrix derivative
%     fprintf('   nonsymmetric derivative case\n')
%     A  = Deriv( Ax, Adx );
%     
%     [R] = chol(A);
%     [Rp] = chol(Ax + step*Adx);
%     rel_error1n = norm( (Rp-Get_value(R))/step - Get_deriv(R) )/norm( Get_deriv(R) );
    
    if ( verbose )
      fprintf('   first  chol test: the relative error is %g\n',rel_error1s);
%       fprintf(' second chol test: the relative error is %g\n',rel_error1n);
    end
    
    if ( rel_error1s > tol ) %|| rel_error1n > tol )
      fprintf('   first  chol test: failed with relative error %g\n',rel_error1s);
%       fprintf(' second chol test: failed with relative error %g\n',rel_error1n);
      success = false;
    end
    
    fprintf('\n')
  end
  
  %  EIG
  %-----------------------------------------------------------------------------
  if (tcase.eig)
    fprintf('   eig:\n')
    n = 5;
    X = randn(n,n); Y = randn(n,n);
    
    % symmetric case (differentiable)
    fprintf('   symmetric case\n')
    A   = Deriv( X+X', Y+Y' ); % subcase 1: symmetric matrix derivative
    Ax  = Get_value(A);
    Adx = Get_deriv(A);
    
    [V,D] = eig(A);
    [Vp,Dp] = eig( Ax + step*Adx );
    rel_error1d = norm( (Dp-Get_value(D))/step - Get_deriv(D) )/norm( Get_deriv(D) );
    rel_error1v = norm( (Vp-Get_value(V))/step - Get_deriv(V) )/norm( Get_deriv(V) );
    
    A   = Deriv( X+X', Y ); % subcase 2: nonsymmetric matrix derivative
    Ax  = Get_value(A);
    Adx = Get_deriv(A);
    
    [V,D] = eig(A);
    [Vp,Dp] = eig( Ax + step*Adx );
    % since Ax+step*Adx is not symmetric, we must sort the eigenvalues for proper fd
    [~,index] = sort( diag(Dp) );
    Vp = Vp(:,index); Dp = Dp(index,index);
    Vx = Get_value(V);
    Vp = Vp*diag( sign(Vx(1,:)./Vp(1,:)) );
    rel_error2d = norm( (Dp-Get_value(D))/step - Get_deriv(D) )/norm( Get_deriv(D) );
    rel_error2v = norm( (Vp-Get_value(V))/step - Get_deriv(V) )/norm( Get_deriv(V) );

    % B=Get_deriv(V)
    % Ax*B+Adx*Get_value(V) - B*Get_value(D) - Get_value(V)*Get_deriv(D)
    if ( verbose )
      fprintf(' first  eig test: the relative error is %g\n',rel_error1d);
      fprintf('                  the relative error is %g\n',rel_error1v);
      fprintf(' second eig test: the relative error is %g\n',rel_error2d);
      fprintf('                  the relative error is %g\n',rel_error2v);
    end
    if ( rel_error1d > tol || rel_error1v > tol )
      fprintf(' first  eig test: failed with relative error %g\n',rel_error1d);
      fprintf('                                             %g\n',rel_error1v);
      success = false;
    end
    if ( rel_error2d > tol || rel_error2v > tol )
      fprintf(' second eig test: failed with relative error %g\n',rel_error2d);
      fprintf('                                             %g\n',rel_error2v);
      success = false;
    end
    
  end
  
  %  LQR
  %-----------------------------------------------------------------------------
  if (tcase.lqr)
    fprintf('   lqr:\n')
    n = 30;
    m = 3; 
    p = 4;
    dd = n*step;
    A = Deriv( rand(n,n), rand(n,n) );
    B = Deriv( rand(n,m), rand(n,m) );
    C = Deriv( rand(p,n), rand(p,n) );
    Q = C'*C;
    R = Deriv( rand(m,m), rand(m,m) );
    R = R'*R;

    N = Deriv( rand(n,m), rand(n,m) );
  
    [K,S,E] = lqr(A,B,Q,R);
  
    [Kp,Sp,Ep] = lqr(Get_value(A)+dd*Get_deriv(A),Get_value(B)+dd*Get_deriv(B),...
                     Get_value(Q)+dd*Get_deriv(Q),Get_value(R)+dd*Get_deriv(R));
    rel_error1 = norm( Get_deriv(K) - ( Kp-Get_value(K) )/dd ) / norm( Get_deriv(K) );
    rel_error2 = norm( Get_deriv(S) - ( Sp-Get_value(S) )/dd ) / norm( Get_deriv(S) );
    if ( verbose )
      fprintf(' first lqr test: the relative error is %g\n',rel_error1);
      fprintf('                 the relative error is %g\n',rel_error2);
    end
    if ( rel_error1 > tol || rel_error2 > tol )
      fprintf(' first lqr test: failed with relative error %g\n',rel_error1);
      fprintf('                                            %g\n',rel_error2);
      success = false;
    end
  end
  
  %  LYAP
  %-----------------------------------------------------------------------------
  if (tcase.lyap)
    fprintf('   lyap:\n')
    n = 4;
    A = Deriv( rand(n,n), rand(n,n) );
    B = Deriv( rand(n,n), rand(n,n) );
    C = Deriv( rand(n,n), rand(n,n) );
    D = Deriv( rand(n,n), rand(n,n) );
  
    X = lyap(A,B);
    Xp = lyap( Get_value(A)+step*Get_deriv(A), Get_value(B)+step*Get_deriv(B) );
    rel_error = norm( Get_deriv(X) - ( Xp-Get_value(X) )/step ) / norm( Get_deriv(X) );
    if ( verbose )
      fprintf(' first lyap test:  the relative error is %g\n',rel_error);
    end
    if ( rel_error > tol )
      fprintf(' first lyap test:  failed with relative error %g\n',rel_error);
      success = false;
    end
  
    X = lyap(A,B,C);
    Xp = lyap( Get_value(A)+step*Get_deriv(A), Get_value(B)+step*Get_deriv(B), ...
               Get_value(C)+step*Get_deriv(C) );
    rel_error = norm( Get_deriv(X) - ( Xp-Get_value(X) )/step ) / norm( Get_deriv(X) );
    if ( verbose )
      fprintf(' second lyap test: the relative error is %g\n',rel_error);
    end
    if ( rel_error > tol )
      fprintf(' second lyap test: failed with relative error %g\n',rel_error);
      success = false;
    end
  
    B  = B+B';   % must be self-adjoint
    X  = lyap(A,B,[],D);
    Xp = lyap( Get_value(A)+step*Get_deriv(A), Get_value(B)+step*Get_deriv(B), [], Get_value(D)+step*Get_deriv(D) );
  
    rel_error = norm( Get_deriv(X) - ( Xp-Get_value(X) )/step ) / norm( Get_deriv(X) );
    if ( verbose )
      fprintf(' third lyap test:  the relative error is %g\n',rel_error);
    end
    if ( rel_error > tol )
      fprintf(' third lyap test:  failed with relative error %g\n',rel_error);
      success = false;
    end
    fprintf('\n')
  end
  
  
  %  HESS
  %-----------------------------------------------------------------------------
  if (tcase.hess)
    fprintf('   hess:\n')
    m  = 5;
    Ax = randn(m,m);  Adx = randn(m,m);
    A  = Deriv( Ax, Adx );
    
    [P,H] = hess(A);
    
    [Pp,Hp] = hess(Ax+step*Adx);
    Hprime = ( Hp-Get_value(H) )/step;
    Pprime = ( Pp-Get_value(P) )/step;
    rel_error1 = norm( Get_deriv(P)'*Get_value(P) + Get_value(P)'*Get_deriv(P) );
    rel_error2 = norm( Pprime-Get_deriv(P) ) / norm( Get_deriv(P) );
    rel_error3 = norm( Hprime-Get_deriv(H) ) / norm( Get_deriv(H) );
    if ( verbose )
      fprintf(' first hess test: the relative error is %g\n',rel_error1);
      fprintf('                  the relative error is %g\n',rel_error2);
      fprintf('                  the relative error is %g\n',rel_error3);
    end
    if ( rel_error1 > tol || rel_error2 > tol || rel_error3 > tol )
      fprintf(' first hess test: failed with relative error %g\n',rel_error1);
      fprintf('                  failed with relative error %g\n',rel_error2);
      fprintf('                  failed with relative error %g\n',rel_error3);
      success = false;
    end
      
    % testing one output argument.
    H = hess(A);
    Hp = hess(Ax+step*Adx);
    Hprime = ( Hp-Get_value(H) )/step;
    rel_error3 = norm( Hprime-Get_deriv(H) ) / norm( Get_deriv(H) );
    if ( rel_error3 > tol )
      fprintf(' second hess test: failed with relative error %g\n',rel_error3);
      success = false;
    end
    
    fprintf('\n')
  end
  
  %  QR
  %-----------------------------------------------------------------------------
  if (tcase.qr)
    fprintf('   qr:\n')
    m = 5;
    n = 5;
    A = Deriv( rand(m,n), rand(m,n) );
    
    [Q,R] = qr(A);
    
    [~,Rp] = qr(Get_value(A)+step*Get_deriv(A));
    Rprime = ( Rp-Get_value(R) )/step;
    rel_error1 = norm( Get_deriv(Q)'*Get_value(Q) + Get_value(Q)'*Get_deriv(Q) );
    rel_error2 = norm( Get_deriv(R) - Rprime) / norm( Get_deriv(R) );
    if ( verbose )
      fprintf(' first qr test: the relative error is %g\n',rel_error1);
      fprintf('                the relative error is %g\n',rel_error2);
    end
    if ( rel_error1 > tol || rel_error2 > tol )
      fprintf(' first qr test: failed with relative error %g\n',rel_error1);
      fprintf('                                           %g\n',rel_error2);
      success = false;
    end
      
    fprintf('\n')
  end
    
  %  SVD
  %-----------------------------------------------------------------------------
  if (tcase.svd)
    fprintf('   svd:\n')
    m = 10;
    n = 7;
    r = min(m,n);
    A = Deriv( rand(m,n), rand(m,n) );
      
    [U,S,V] = svd(Get_value(A));
    [Up,Sp,Vp] = svd(Get_value(A)+1e-8*Get_deriv(A));
    Ud = (Up-U)/1e-8; Sd = (Sp-S)/1e-8; Vd = (Vp-V)/1e-8;
    
    rel_error1 = abs( (U(:,1)'*Get_deriv(A)*V(:,1)) - Sd(1,1) )/abs( Sd(1,1) );
    rel_error2 = abs( (U(:,2)'*Get_deriv(A)*V(:,2)) - Sd(2,2) )/abs( Sd(2,2) );
    rel_error3 = abs( (U(:,r)'*Get_deriv(A)*V(:,r)) - Sd(r,r) )/abs( Sd(r,r) );
    if ( verbose )
      fprintf(' the first svd test: the relative error is %g\n',rel_error1);
      fprintf('                                           %g\n',rel_error2);
      fprintf('                                           %g\n',rel_error3);
    end
    if ( rel_error1 > tol || rel_error2 > tol || rel_error3 > tol )
      fprintf(' first svd test: failed with relative error %g\n',rel_error1);
      fprintf('                                            %g\n',rel_error2);
      fprintf('                                            %g\n',rel_error3);
      success = false;
    end

    fprintf('\n')
  end
  
  %  ABS
  %-----------------------------------------------------------------------------
  if (tcase.abs)
    fprintf(' Testing abs()\n')
    n = 10;
    c = Deriv( rand(n,1)-.5, rand(n,1)-.5 );
    c = abs(c);
    
    rel_error1 = norm( ( abs(Get_value(c)+1e-7*Get_deriv(c))-abs(Get_value(c)))/1e-7 - Get_deriv(c) )/norm( abs(Get_value(c)) );
%     ( abs(Get_value(c)+1e-7*Get_deriv(c))-abs(Get_value(c)))/1e-7
%     Get_deriv(c)
    
    if ( verbose )
      fprintf(' the first abs test: the relative error is %g\n',rel_error1);
    end
    if ( rel_error1 > tol )
      fprintf(' first abs test: failed with relative error %g\n',rel_error1);
      success = false;
    end
    
    fprintf('\n')
  end
  
  
  %% ---------------------------------------------------------------------------
  %  Testing Matrix Functions
  %  NORM
  %-----------------------------------------------------------------------------
  if (tcase.norm)
    fprintf('   norm: matrix cases\n')
    n = 5;
    A = Deriv( rand(n,n), rand(n,n) );
    Ap = Get_value(A)+step*Get_deriv(A);

    x = norm(A,'fro');
    rel_error = abs(( (norm(Ap,'fro')-Get_value(x) )/step - Get_deriv(x) )/ Get_deriv(x));
    if ( verbose )
      fprintf(' Matrix ''fro'' norm: relative error is %g\n',rel_error);
    end
    if ( rel_error > tol )
      fprintf(' Matrix ''fro'' norm:  failed with relative error %g\n',rel_error);
      success = false;
    end
    
    x = norm(A,1);
    rel_error = abs(( (norm(Ap,1)-Get_value(x) )/step - Get_deriv(x) )/ Get_deriv(x));
    if ( verbose )
      fprintf(' Matrix     1 norm: relative error is %g\n',rel_error);
    end
    if ( rel_error > tol )
      fprintf(' Matrix     1 norm:  failed with relative error %g\n',rel_error);
      success = false;
    end
    
    x = norm(A,2);
    rel_error = abs(( (norm(Ap,2)-Get_value(x) )/step - Get_deriv(x) )/ Get_deriv(x));
    if ( verbose )
      fprintf(' Matrix     2 norm: relative error is %g\n',rel_error);
    end
    if ( rel_error > tol )
      fprintf(' Matrix     2 norm:  failed with relative error %g\n',rel_error);
      success = false;
    end

    x = norm(A,inf);
    rel_error = abs(( (norm(Ap,inf)-Get_value(x) )/step - Get_deriv(x) )/ Get_deriv(x));
    if ( verbose )
      fprintf(' Matrix   inf norm: relative error is %g\n',rel_error);
    end
    if ( rel_error > tol )
      fprintf(' Matrix   inf norm:  failed with relative error %g\n',rel_error);
      success = false;
    end

    fprintf('   norm: vector cases\n')
    v  = Deriv( randn(n,1), randn(n,1) );
    vp = Get_value(v) + step*Get_deriv(v);
    
    x = norm(v,1);
    rel_error = abs(( (norm(vp,1)-Get_value(x) )/step - Get_deriv(x) )/ Get_deriv(x));
    if ( verbose )
      fprintf(' Vector    1 norm: relative error is %g\n',rel_error);
    end
    if ( rel_error > tol )
      fprintf(' Vector    1 norm:  failed with relative error %g\n',rel_error);
      success = false;
    end
    
    x = norm(v,2);
    rel_error = abs(( (norm(vp,2)-Get_value(x) )/step - Get_deriv(x) )/ Get_deriv(x));
    if ( verbose )
      fprintf(' Vector    2 norm: relative error is %g\n',rel_error);
    end
    if ( rel_error > tol )
      fprintf(' Vector    2 norm:  failed with relative error %g\n',rel_error);
      success = false;
    end
    
    p = 4.5;
    x = norm(v,p);
    rel_error = abs(( (norm(vp,p)-Get_value(x) )/step - Get_deriv(x) )/ Get_deriv(x));
    if ( verbose )
      fprintf(' Vector    p norm: relative error is %g\n',rel_error);
    end
    if ( rel_error > tol )
      fprintf(' Vector    p norm:  failed with relative error %g\n',rel_error);
      success = false;
    end
    
    x = norm(v,Inf);
    rel_error = abs(( (norm(vp,Inf)-Get_value(x) )/step - Get_deriv(x) )/ Get_deriv(x));
    if ( verbose )
      fprintf(' Vector  Inf norm: relative error is %g\n',rel_error);
    end
    if ( rel_error > tol )
      fprintf(' Vector  Inf norm:  failed with relative error %g\n',rel_error);
      success = false;
    end

    x = norm(v,-Inf);
    rel_error = abs(( (norm(vp,-Inf)-Get_value(x) )/step - Get_deriv(x) )/ Get_deriv(x));
    if ( verbose )
      fprintf(' Vector -Inf norm: relative error is %g\n',rel_error);
    end
    if ( rel_error > tol )
      fprintf(' Vector -Inf norm:  failed with relative error %g\n',rel_error);
      success = false;
    end

    fprintf('\n')
  end
  
  %%----------------------------------------------------------------------------
  %  INTERP1
  %-----------------------------------------------------------------------------
  if (tcase.interp1)
    fprintf(' Testing interpolation operations\n')
    fprintf('   interp1:\n')
    x = linspace(0,pi,20);
    y = Deriv( cos(x), -sin(x) );
    xi = linspace(0,pi,5);
    
    % nearest neighbor case (note that finite difference fails if 
    % x = linspace(0,pi,20); and xi = linspace(0,pi,5); above)
    yi = interp1(x,y,xi,'nearest');
    yip = interp1(x,Get_value(y)+step*Get_deriv(y),xi,'nearest');
    rel_error1 = norm( (yip-Get_value(yi))/step - Get_deriv(yi) )/norm( Get_deriv(yi) );
    if ( verbose )
      fprintf('    nearest test: the relative error is %g\n',rel_error1);
    end
    
    % linear case (note that finite difference fails if 
    % x = linspace(0,pi,20); and xi = linspace(0.pi,5); above)
    yi = interp1(x,y,xi,'linear');
    yip = interp1(x,Get_value(y)+step*Get_deriv(y),xi,'linear');
    rel_error2 = norm( (yip-Get_value(yi))/step - Get_deriv(yi) )/norm( Get_deriv(yi) );
    if ( verbose )
      fprintf('     linear test: the relative error is %g\n',rel_error2);
    end
    
    % spline case
    yi = interp1(x,y,xi,'spline');
%     Get_value(yi)
%     interp1(x,Get_value(y),xi,'spline')
    yip = interp1(x,Get_value(y)+1000000*step*Get_deriv(y),xi,'spline');
%     Get_deriv(yi)
%     (yip-Get_value(yi))/(1000000*step)
%     norm( (yip-Get_value(yi))/(1000000*step) - Get_deriv(yi) )
    rel_error3 = norm( (yip-Get_value(yi))/(1000000*step) - Get_deriv(yi) )/norm( Get_deriv(yi) );
    if ( verbose )
      fprintf('     spline test: the relative error is %g\n',rel_error3);
    end
    
    fprintf('\n')    
  end
  
  %-----------------------------------------------------------------------------
  %  Testing ode/dae solvers
  %-----------------------------------------------------------------------------
  if (tcase.ode)
    fprintf(' Testing ode solvers\n')
    
    %  ODE23
    %---------------------------------------------------------------------------
    fprintf(' ode23:\n')
    tspan = [0,1];
%     y0   = Deriv(3,1);
%     ydot = @(t,y)(3*y);
%     [T,Y] = ode23(ydot,tspan,y0);
%     plot(T,Get_value(Y),'+',T,3*exp(3*T),'-')
%     hold on
%     plot(T,Get_deriv(Y),'b*',T,exp(3*T),'-')
%     pause
    
%    a = Set_variable(4);
%    y0 = a;
%    ydot = @(t,y)(a*y);
%    [T,Y] = ode23(ydot,tspan,y0);
%    plot(T,Get_value(Y),'+',T,Get_value(a)*exp(Get_value(a)*T))
%    hold on
%    plot(T,Get_deriv(Y),'b+',T,exp(Get_value(a)*T)+Get_value(a)^2*exp(Get_value(a)*T))
%    pause
    
     y0   = Deriv(.999,.0000001);
%    y0   = Deriv(.999,1);
    y0x  = Get_value(y0);
    y0dx = Get_deriv(y0);
    ydot = @(t,y)(y.^2);
    
    [T,Y] = ode23(ydot,tspan,y0);
    [Tp,Yp] = ode23(ydot,tspan,y0x+step*y0dx);
    Yp_dot = feval(ydot,Tp,Yp);
    Yp = Yp - Yp_dot.*(Tp-T); 
    fd = ( Yp-Get_value(Y) )/step;
    Sexact = ( y0x*T./(1-y0x*T).^2 + 1./(1-y0x*T) )*y0dx;
    semilogy(T,Get_deriv(Y),'b+',T,fd,'r^',T,Sexact,'k-')
    legend('Deriv','Finite Differences','Exact')
    title('Differentiating ode23')
    xlabel('x'); ylabel('\partial y/\partial y_0')
    fprintf('\n')    
%     figure
%     hold on
%     plot(Tp,'ko')
%     plot(T,'b+')
    
    fprintf(' ode45:\n')
    figure
    [T ,Y ] = ode45(ydot,tspan,y0           );
    [Tp,Yp] = ode45(ydot,tspan,y0x+step*y0dx);
    Yp_dot = feval(ydot,Tp,Yp);
    Yp = Yp - Yp_dot.*(Tp-T); 
    fd = ( Yp-Get_value(Y) )/step;
    Sexact = ( y0x*T./(1-y0x*T).^2 + 1./(1-y0x*T) )*y0dx;
    semilogy(T,Get_deriv(Y),'b+',T,fd,'r^',T,Sexact,'k-')
    legend('Deriv','Finite Differences','Exact')
    title('Differentiating ode45')
    xlabel('x'); ylabel('\partial y/\partial y_0')
    fprintf('\n')    
%     figure
%     hold on
%     plot(Tp,'ko')
%     plot(T,'b+')
    
    
  end
  
  if ( success )
    fprintf(' Deriv passed all of the following tests\n')
    if (tcase.arithmetic), fprintf('  arithmetic\n'); end 
    if (tcase.eig       ), fprintf('  eig\n');        end
    if (tcase.interp1   ), fprintf('  interp1\n');    end
    if (tcase.lqr       ), fprintf('  lqr\n');        end 
    if (tcase.lyap      ), fprintf('  lyap\n');       end 
    if (tcase.norm      ), fprintf('  norm\n');       end 
    if (tcase.qr        ), fprintf('  qr\n');         end 
    if (tcase.svd       ), fprintf('  svd\n');        end 

  end
end

