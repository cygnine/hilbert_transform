function[h] = hilbert_eno_periodic(theta,f,varargin)
% hilbert_eno_periodic -- Hilbert Transform based on ENO reconstruction
%
% h = hilbert_eno(theta,f,{phi=false, interval=[0,2*pi], k=3, Nq=10})
%
%     Computes the Hilbert Transform of the point values f. The input data is
%     (theta,f), and the Hilbert Transform is computed at the locations phi. If
%     phi is not given, it is assumed to be the same as theta. The function
%     given the point values f is assumed to be periodic over the optional input
%     interval. The default interval assumed is [0, 2*pi). It is assumed that
%     theta is sorted in increasing order.
%
%     Uses a k-th order ENO periodic reconstruction on each cell between point
%     values and uses an Nq-point Legendre-Gauss quadrature rule on each cell.
%     On the cell where the singularity of the Hilbert Kernel lies, the
%     principal value is computed via the Legendre modal coefficients.
%
%     Due to the use of ENO, this method is not linear, but if you modulo out
%     the nonlinear choice of stencil, the other operations are all linear.

global packages;
inputs = {'phi', 'k', 'Nq', 'interval'};
defaults = {false, 3, 10, [0, 2*pi]};
opt = packages.labtools.input_schema(inputs, defaults, [], varargin{:});

gq = packages.speclab.orthopoly1d.jacobi.quad.gauss_quadrature.handle;
eno = packages.eno.eno_interpolant_periodic.handle;
polys = packages.speclab.orthopoly1d.jacobi.eval.eval_jacobi_poly.handle;

theta = theta(:);
tol = 1e-12;

leg_opt.alpha = 0;
leg_opt.beta = 0;
[r,w] = gq(opt.Nq, leg_opt);

if not(opt.phi)
  opt.phi = theta;
  error('Not coded yet: define the optional input phi');
else
  % If there are values of phi that match up with theta, we'll have to be careful
  phi_size= size(opt.phi);
  opt.phi = opt.phi(:);
end

if mod(opt.Nq,2)==1
  warning('With an odd # of grid points per cell, the cot function will be evaluated with argument 0');
end

N = length(theta);
M = length(opt.phi);
[garbage, bin] = histc(opt.phi, theta);
bin(bin==0) = N;  % These lie inside the periodic extension [interval(2), interval(1)]

% For the sake of this code, put periodic theta(1) on other side as well:
theta(end+1) = opt.interval(2) + (theta(1) - opt.interval(1));

% The hard part: find the vertices of the quadrature grid
[all_grid, indices] = sort([theta; opt.phi]);
new_grid = theta(1);
% For each phi: find left and right bounding cells
left_lengths = opt.phi - theta(bin);
right_lengths = theta(bin+1) - opt.phi;
if any(abs(left_lengths)<tol | abs(right_lengths)<tol)
  error('This code doesn''t yet support evaluating the Hilbert Transform at the same grid points as the input');
end
midpoints = abs(right_lengths-left_lengths)<tol;
lean_left = left_lengths<right_lengths & not(midpoints);
lean_right = left_lengths>right_lengths & not(midpoints);

singular_cell = false([M 1]);  % false = left, true = right
left_cell = zeros([M 2]);
left_cell(:,1) = theta(bin);
left_cell(midpoints,2) = theta(bin(midpoints)+1);
left_cell(lean_left,2) = theta(bin(lean_left)) + 2*left_lengths(lean_left);
left_cell(lean_right,2) = theta(bin(lean_right)+1) - 2*right_lengths(lean_right);

singular_cell = lean_right;

right_cell_needed = true([M, 1]);
right_cell_needed(midpoints) = false;

% Ok, now generate global common vertices
cell_scale = diff(theta.')/2;
cell_shift = (theta(2:end).' + theta(1:(end-1)).')/2;
vertices = repmat(r, [1, N])*spdiags(cell_scale(:), 0, N,N);
vertices = vertices + repmat(cell_shift, [opt.Nq 1]);

% Shift extended periodic modes back:
vertices(:,end) = mod(vertices(:,end)-opt.interval(1), diff(opt.interval)) + opt.interval(1);

% Now we can do quadrature everywhere:
H = repmat(opt.phi, [1, opt.Nq*N]) - repmat(vertices(:).', [M 1]);
H = cot(H/2);
tempw = repmat(w,[1,N])*spdiags(cell_scale.',0,N,N);
H = H*spdiags(tempw(:),0,N*opt.Nq, N*opt.Nq);
% Don't do quadrature over singular intervals
for q = 1:M;
  i1 = 1 + (bin(q)-1)*opt.Nq;
  i2 = bin(q)*opt.Nq;
  H(q,i1:i2) = 0;
end

% eno-interpolate the data to the vertices:
f_vertices = eno(theta(1:end-1), f, vertices, opt.interval, 'k', opt.k);

% The `non-singular' Hilbert transform contribution:
h = H*f_vertices(:);

% For the singular part, first we'll need to interpolate f to all the broken
% cell locations:
left_scale = diff(left_cell,[],2)/2;
left_shift = mean(left_cell,2);
right_scale = zeros([M 1]);
vertices = zeros([opt.Nq*M + sum(right_cell_needed)*opt.Nq 1]);
bin_indices = zeros([M 2]);
for q = 1:M
  if q==1
    bin_indices(q,1) = 1;
  else
    bin_indices(q,1) = 1 + bin_indices(q-1,2);
  end
  i1 = bin_indices(q,1); 
  i2 = i1 + opt.Nq - 1;
  vertices(i1:i2) = r*left_scale(q)+left_shift(q);
  if right_cell_needed(q)
    right_scale(q) = (theta(bin(q)+1) - left_cell(q,2))/2;
    right_shift = (theta(bin(q)+1) + left_cell(q,2))/2;
    bin_indices(q,2) = 2*opt.Nq + bin_indices(q,1) - 1;
    i1 = i2 + 1;
    i2 = i1 + opt.Nq - 1;
    vertices(i1:i2) = r*right_scale(q)+right_shift;
  else
    bin_indices(q,2) = opt.Nq + bin_indices(q,1) - 1;
  end
end


% eno-interpolate the data to the vertices:
f_vertices = eno(theta(1:end-1), f, vertices, opt.interval, 'k', opt.k);

% Now add in all the `left' singular contributions:
indices = repmat((1:opt.Nq).'-1, [1 M]);
indices = ones([opt.Nq, M])*spdiags(bin_indices(:,1),0,M,M) + indices;
x = repmat(opt.phi.',[opt.Nq 1]) - vertices(indices);
contribution = left_scale.*(w'*(f_vertices(indices).*cot(x/2))).';

% And the `right' singular contributions, if applicable
Mr = sum(right_cell_needed);
indices = repmat((1:opt.Nq).' + opt.Nq - 1, [1 Mr]);
indices = ones([opt.Nq, Mr])*spdiags(bin_indices(right_cell_needed,1),0,Mr,Mr) + indices;
x = repmat(opt.phi(right_cell_needed).', [opt.Nq 1]) - vertices(indices);
contribution(right_cell_needed) = contribution(right_cell_needed) + ...
   right_scale(right_cell_needed).*(w'*(f_vertices(indices).*cot(x/2))).';

h = h + contribution;

% Finally:
h = h/(2*pi);

h = reshape(h,phi_size);
