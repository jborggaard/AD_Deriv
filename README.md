*Deriv* - A set of Matlab tools for Automatic Differentiation
=======


 
Overview
--------

This directory contains a set of tools for performing automatic differentiation
in Matlab.  Automatic differentiation is performed using Matlab's operator
overloading.  The intention is to replace *double* objects with *Deriv* objects
(which contain both values of the variable (acting as a double) as well as the
derivative of the variable with respect to the parameter(s) of interest.
To perform the derivative calculations in Matlab, Deriv must overload any of 
the *double* class operators that Matlab encounters when executing your
function.  The value of the derivative of any intermediate calculation can
be extracted using the _Get_deriv_ function.  

A differentiating feature of *Deriv* is that it goes back to the continuous 
form of a problem to compute derivatives of non-differentiable objects.  For
example, we avoid differentiation of the time-step selection process in *ode23*
by simultaneously computing the solution of the original and differentiated
ode in our own algorithm.  This includes new algorithms for implementation 
of Matlab functions such as *interp1*, *QR*, etc., which would be approximated by
finite differences in other automatic differentiation algorithms.

Repository Contents
--------
* _Deriv_  A Matlab class (this can stand alone and must be in your path) that implements automatic differentiation by operator overloading.  All of the overloaded _double_ functions are contained in this file.

* _Dzeros_  A Matlab function that takes care of the preallocation problem.  Any preallocated variables in your differentiated function that depend on the independent variable must be replaced with this Dzeros function for now.  A simple global change and replace zeros->Dzeros would work, though more efficiency can be introduced by selectively replacing only those functions that are affected by the independent variable.

* _Get_gradient_  A Matlab function that uses the forward mode of automatic differentiation to compute the gradient of a function.

* _LICENSE.md_  The LGPL license.

* _README.md_  This file.

* _Set_variable_  A Matlab function that defines the independent variable for automatic differentiation.

* _test_Deriv_  A series of unit tests for _Deriv_.


Basic Introduction
--------

To illustrate how this works, we use the function *Set_variable* to define
the variable used as the independent variable.  Any number of intermediate
calculations (with or without using alpha) can be performed.  The derivative
of those calculations with respect to alpha can be extracted with the 
embedded *Get_deriv* method (hidden inside *Deriv*).

```matlab
    >> alpha = Set_variable(7);
    >>
    >> % Perform intermediate calculations to arrive at the desired output.
    >>
    >> output = exp((alpha-5)^3)*sin(alpha);
    >> f = Get_value(output);  % extracts the value of the output
    >> d = Get_deriv(output);  % extracts the derivative of the output wrt alpha
    >>
    >> disp(f)
       1.9584e+03
   
    >> disp(d)
       2.5749e+04
````

Development Tasks
--------
- [x] overload basic double operators
- [x] overload matrix functions
- [ ] overload ode/dae solvers
- [ ] overload interp options

- [x] scalar variable, forward mode
- [ ] vector variable, forward mode
- [ ] vector variable, reverse mode
- [ ] Jacobian-vector product, reverse mode

- [ ] user documentation (this README.md)
- [ ] algorithm documentation (paper, wiki)


Author
--------
Jeff Borggaard, Interdisciplinary Center for Applied Mathematics, Virginia Tech
jborggaard@vt.edu

License
--------
These files are provided under the Gnu LGPL License.
