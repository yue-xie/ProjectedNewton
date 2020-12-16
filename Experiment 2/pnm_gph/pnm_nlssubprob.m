function [H,GRAD,iter] = pnm_nlssubprob(V,W,Hinit,tol,maxiter)
% Solve nonnegative least squares subproblem:
% min_H 1/2*||V - WH||_F^2  s.t. H_ij >= 0

% ----------------- INPUT ---------------------------
% V, W: nonnegative constant matrices
% Hinit: initial solution
% tol: stopping tolerance
% maxiter: limit of iterations

% ----------------OUTPUT ---------------------------
% H: output solution 
% GRAD: output gradient
% iter: #iterations used

% Written by Pinghua Gong, Tsinghua University
% Part of this function is modified based on Prof. Chih-Jen Lin's nlssubprob.m


H = Hinit; 
WtV = W'*V;
WtW = W'*W;
[r,m] = size(H);

sigma = 1e-3; beta = 5e-1; epsilon = 1e-12;
for iter=1:maxiter,
  GRAD = WtW*H - WtV;
  projgrad = norm(GRAD(GRAD < 0 | H >0));
  if projgrad < tol,
    break;
  end
  
  epsilonk = min(epsilon,norm(H - max(0,H - GRAD),'fro'));
  INDk = ((H >= 0) & (H <= epsilonk) & (GRAD > 0)); invINDk = xor(ones(r,m),INDk);
  Pk = computePk(WtW,GRAD,INDk,invINDk,r,m);

  % search step size
  mk = 0;
  for inner_iter=1:20,
     Hn = max(0,H - beta^mk*Pk); dH = Hn - H;
     gradd=sum(sum(GRAD.*dH)); dQd = sum(sum((WtW*dH).*dH));
     if gradd + 0.5*dQd <= -sigma*(beta^mk*sum(sum(GRAD(invINDk).*Pk(invINDk))) - sum(sum(dH(INDk).*GRAD(INDk))))
        break;
     else
        mk = mk + 1;
     end 
  end
  H = Hn;
      
end

if iter==maxiter,
  fprintf('Max iter in pnm_nlssubprob\n');
end