function [W,H,iter,time,timeaxis,objstr] = pnm_nmf(V,Winit,Hinit,tol,timelimit,maxiter)
% Solve nonnegative matrix factorization problem:
% min_W,H 1/2*||V - WH||_F^2  s.t. W_ij >= 0, H_ij >= 0

% ------------------ INPUT ------------------------
% V: nonnegative constant matrix
% Winit,Hinit: initial solution
% tol: tolerance for a relative stopping condition
% timelimit, maxiter: limit of time and iterations

% ----------------- OUTPUT ------------------------
% W,H: output solution

% Written by Pinghua Gong, Tsinghua University
% Part of this function is modified based on Prof. Chih-Jen Lin's nmf.m

W = Winit; H = Hinit; tic; etime = 0;

gradW = W*(H*H') - V*H'; gradH = (W'*W)*H - W'*V;
temp = toc;
initgrad = norm([gradW; gradH'],'fro');
fprintf('Init gradient norm %f\n', initgrad);
etime = etime + toc - temp;
tolW = max(0.001,tol);%*initgrad; Yue changed this to absolute error
tolH = tolW;
objstr = [];
timeaxis = [];
projnorm = norm([gradW(gradW<0 | W>0); gradH(gradH<0 | H>0)]);

for iter=1:maxiter,
    if toc > timelimit | projnorm < tol,%Yue changed the stopping criterion
        break;
    end
    
    [W,gradW,iterW] = pnm_nlssubprob(V',H',W',tolW,1000); W = W'; gradW = gradW';
    if iterW==1,
        tolW = 0.1 * tolW; 
    end
    
    [H,gradH,iterH] = pnm_nlssubprob(V,W,H,tolH,1000);
    if iterH==1,
        tolH = 0.1 * tolH;
    end
    
    temp = toc;
    objstr = [objstr;.5*norm(V - W*H,'fro')^2];
    etime = etime + toc - temp;
    timeaxis = [timeaxis,toc - etime];
    gradW = W*(H*H') - V*H';
    projnorm = norm([gradW(gradW<0 | W>0); gradH(gradH<0 | H>0)]);
end
time = toc - etime;
fprintf('\nFinal Iteration %d\n', iter);
