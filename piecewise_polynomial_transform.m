function[h] = piecewise_polynomial_transform(piece_poly, x, varargin)
% piecewise_polynomial_transform -- Hilbert transform from piecewise polynomial
%
% [h] = piecewise_polynomial_transform(piece_poly, x, {cells, alpha=0, beta=0, Nq=2*(N+1)})
%
%     Computes the Hilbert transform of a piecewise polynomial function and
%     evaluates it at the points x. No points of x may coincide with the knots
%     defining the piecewise function; an error is thrown in this case.
%
%     The input piece_poly is an N x K matrix of coefficients. piece_poly(:,k)
%     are the N modal coefficients for the first N (local) Jacobi(alpha,beta)
%     polynomials on cell k. The cell boundaries are given by the (K+1)-length
%     vector cells, which is mandatory. Nq is the number of quadrature nodes
%     used on each interval. 
%
%     This function assumes that interval of approximation is [0,2*pi].

[N,K] = size(piece_poly);

global handles;
inputs = {'cells', 'alpha', 'beta', 'Nq'};
defaults = {[], 0, 0, 2*(N+1)};
opt = handles.common.input_schema(inputs, defaults, [], varargin{:});

cot_expansion = handles.hilbert_transform.cotangent_coefficients.handle;
opoly = handles.speclab.orthopoly1d;
jac = handles.speclab.orthopoly1d.jacobi;
gq = jac.quad.gauss_quadrature.handle;
evalpoly = opoly.eval_polynomial_standard.handle;

if isempty(opt.cells)
  error('You must define the cells over which the piecewise polynomial is defined');
end

Nq = opt.Nq;
x_size = size(x);
x = x(:);

M = length(x);

% Get Jacobi polynomial cotangent expansion:
cotangent_coeffs = cot_expansion('alpha', opt.alpha, 'beta', opt.beta);

[garbage, bin] = histc(x, opt.cells);
bin(x==opt.cells(end)) = K;

% The hard part: find the vertices of the quadrature grid
tol = 1e-12; % if an x is less than this distance from a knot, error
[all_grid, indices] = sort([opt.cells; x]);
new_grid = opt.cells(1);

% For each x: find left and right bounding cells
left_lengths = x - opt.cells(bin);
right_lengths = opt.cells(bin+1) - x;
if any(abs(left_lengths)<tol | abs(right_lengths)<tol)
  error('This code doesn''t yet support evaluating the Hilbert Transform at the same grid points as the input');
end
midpoints = abs(right_lengths-left_lengths)<tol;
lean_left = left_lengths<right_lengths & not(midpoints);
lean_right = left_lengths>right_lengths & not(midpoints);

singular_cell = false([M 1]);  % false = left, true = right
left_cell = zeros([M 2]);
left_cell(:,1) = opt.cells(bin);
left_cell(midpoints,2) = opt.cells(bin(midpoints)+1);
left_cell(lean_left,2) = opt.cells(bin(lean_left)) + 2*left_lengths(lean_left);
left_cell(lean_right,2) = opt.cells(bin(lean_right)+1) - 2*right_lengths(lean_right);

singular_cell = lean_right;

right_cell_needed = true([M, 1]);
right_cell_needed(midpoints) = false;

% Ok, now generate global common vertices
jopt.alpha = 0;
jopt.beta = 0;
[r,w] = gq(Nq,jopt);

cell_scale = diff(opt.cells.')/2;
cell_shift = (opt.cells(2:end).' + opt.cells(1:(end-1)).')/2;
vertices = repmat(r, [1, K])*spdiags(cell_scale(:), 0, K,K);
vertices = vertices + repmat(cell_shift, [Nq 1]);

% Now quadrature can be done on each cell:
H = repmat(x, [1, Nq*K]) - repmat(vertices(:).', [M 1]);
H = cot(H/2);
tempw = repmat(w,[1,K])*spdiags(cell_scale.',0,K,K);
H = H*spdiags(tempw(:),0,K*Nq, K*Nq);

% Don't do quadrature over singular intervals
for q = 1:M;
  i1 = 1 + (bin(q)-1)*Nq;
  i2 = bin(q)*Nq;
  H(q,i1:i2) = 0;
end

% Evaluate the function at the nodes:
jopt.alpha = opt.alpha;
jopt.beta = opt.beta;
[recurrence_a, recurrence_b] = jac.coefficients.recurrence(N,jopt);
standard_polys = evalpoly(r,recurrence_a, recurrence_b,0:(N-1));
f_vertices = standard_polys*piece_poly;

% The `non-singular' Hilbert transform contribution:
h = H*f_vertices(:);

% For the singular part, each cell with the singularity is broken into a two
% smaller cells: one of the cells symmetrically surrounds the singularity, and
% the other is the remaining portion of the cell whose contribution can be
% determined via quadrature.

% Do quadrature over the nonsingular cell: this is only necessary when
% right_cell_needed is true
left_scale = diff(left_cell,[],2)/2;
left_shift = mean(left_cell,2);
right_scale = (opt.cells(bin+1) - left_cell(:,2))/2;
%right_scale = diff(opt.cells(bin+1) - left_cell(:,2))/2;
right_shift = mean([opt.cells(bin+1), left_cell(:,2)],2);

% These are the vertices local to each cell within the nonsingular cell
left_vertices = repmat(r, [1,M])*spdiags(left_scale,0,M,M);
left_vertices = left_vertices + repmat(left_shift' - cell_shift(bin), [Nq, 1]);
left_vertices = left_vertices*spdiags(1./cell_scale(bin).',0,M,M);

right_vertices = repmat(r, [1,M])*spdiags(right_scale,0,M,M);
right_vertices = right_vertices + repmat(right_shift' - cell_shift(bin), [Nq, 1]);
right_vertices = right_vertices*spdiags(1./cell_scale(bin).',0,M,M);

% Compute contribution from all singular cells
for q = 1:M

  if singular_cell(q)
    nonsingular_vertices = left_vertices(:,q);
    nonsingular_scale = left_scale(q);
    singular_vertices = right_vertices(:,q);
    singular_scale = right_scale(q);
  else
    nonsingular_vertices = right_vertices(:,q);
    nonsingular_scale = right_scale(q);
    singular_vertices = left_vertices(:,q);
    singular_scale = left_scale(q);
  end

  % 'nonsingular' cell:
  if right_cell_needed(q)
    polys = evalpoly(nonsingular_vertices,recurrence_a, recurrence_b, 0:(N-1));
    fx = polys*piece_poly(:,bin(q));
    vertices = nonsingular_vertices*cell_scale(bin(q)) + cell_shift(bin(q));
    h(q) = h(q) + nonsingular_scale*sum(w.*fx.*cot((x(q)-vertices)/2));
  end
  
  % For the singular subcell, get modal expansion of f(x):
  polys = evalpoly(singular_vertices,recurrence_a, recurrence_b, 0:(N-1));
  fx = polys*piece_poly(:,bin(q));
  f_modes = standard_polys'*spdiags(w,0,Nq,Nq)*fx;

  % Polynomial contribution:
  h(q) = h(q) + singular_scale*sum(f_modes.*cotangent_coeffs(1:N));

  % 2/x contribution:
  singular_vertices = singular_vertices*cell_scale(bin(q)) + cell_shift(bin(q));
  temp = spdiags(2./(x(q)-singular_vertices),0,Nq,Nq)*standard_polys;
  modal_contributions = w'*temp;
  h(q) = h(q) + singular_scale*dot(modal_contributions(2:2:end),f_modes(2:2:end));
end

h = h/(2*pi);
