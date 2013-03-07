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
* _Deriv_  A Matlab class (this can stand alone, and must be in your path)
* Get_value 


Basic Introduction
--------

To illustrate how this works, we use the function *Set_variable* to define
the variable used as the independent variable.  Any number of intermediate
calculations (with or without using alpha) can be performed.  The derivative
of those calculations with respect to alpha can be extracted with the 
embedded *Get_deriv* method (hidden inside *Deriv*)
.
> >> alpha = Set_variable(7);
> >>
> >> %  Perform intermediate calculations to arrive at the desired output.
> >> output = exp((alpha-5)^3)*sin(alpha);
> >> f = Get_value(output);   % extracts the value of the output
> >> d = Get_deriv(output);   % extracts the derivative of the output wrt alpha
> >>
> >> disp(f)
>    1.9584e+03
>
> >> disp(d)
>    2.5749e+04
>

Author
--------
Jeff Borggaard, Interdisciplinary Center for Applied Mathematics, Virginia Tech
> jborggaard@vt.edu

License
--------
These files are provided under the Gnu LGPL License.
