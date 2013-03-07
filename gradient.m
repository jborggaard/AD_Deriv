function g = gradient (x)

  g(1) = -400*x(1)*(x(2)-x(1)^2) - 2*(1-x(1));

  g(2) = 200*(x(2)-x(1)^2) + 20.2*(x(2)-1) + 19.8*(x(4)-1);

  g(3) = -360*x(3)*(x(4)-x(3)^2) -2*(1-x(3));

  g(4) = 180*(x(4)-x(3)^2) + 20.2*(x(4)-1) + 19.8*(x(2)-1);