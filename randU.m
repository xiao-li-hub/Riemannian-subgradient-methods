function U = randU(n)

% U = RANDU(N) generates a random N x N orthogonal matrix, distributed uniformly according to the Haar measure.
%
% Reference:
%
% Mezzadri, Francesco. "How to generate random matrices from the classical compact groups."
% arXiv preprint math-ph/0609050 (2006).

  X = randn(n);
  [Q,R] = qr(X);
  r = sign(diag(R));
  U = bsxfun(@times, Q, r');
