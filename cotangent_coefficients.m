function[modes] = cotangent_coefficients(varargin)
% cotangent_coefficients - Jacobi polynomial approximation for cot(x/2)
%
% modes = cotangent_coefficients({alpha=0, beta=0})
%
%     Computes the first few modal coefficients for a polynomial expansion of
%     cot(x/2) - 2/x. Uses the exact Taylor Series and connects the monomial
%     coefficients to Jacobi polynomial coefficients centered at 0, with default
%     scaling. 

global handles;
jopt = handles.common.input_schema({'alpha', 'beta'}, {0,0}, [], varargin{:});
jac = handles.speclab.orthopoly1d.jacobi;

% The first few Bernoulli numbers, which define the Taylor series coefficients
B = zeros([19 1]);
B(1) = 1;
B(2) = -1/2;
B(3) = 1/6;
B(5) = -1/30;
B(7) = 1/42;
B(9) = -1/30;
B(11) = 5/66;
B(13) = -691/2730;
B(15) = 7/6;
B(17) = -3617/510;
B(19) = 43867/798;

T = zeros([floor(length(B)/2) 1]);  % The Taylor coefficients
for q = 1:(floor(length(B)/2))
  T(q) = (-1)^q*2*B(2*q+1)/factorial(2*q);
end

% Ok, let's find Jacobi coefficients
jopt.scale = 1; % Everything's a polynomial, scaling doesn't matter

[x,w] = jac.quad.gauss_quadrature(2*length(B),jopt);
polys = jac.eval.eval_jacobi_poly(x,0:(length(B)-1),jopt);

taylor_approx = 0;  % 1/x is implicitly assumed
for q = 1:(length(B)/2)
  taylor_approx = taylor_approx + T(q)*x.^(2*q-1);
end

% Use (exact) quadrature to determine modal coefficients
modes = polys'*spdiags(w,0,2*length(B),2*length(B))*taylor_approx;
modes(1:2:end) = 0;
