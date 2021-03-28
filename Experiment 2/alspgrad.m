function [W,H,iter,time,timeaxis,objstr] = alspgrad(V,Winit,Hinit,tol,timelimit,maxiter)

% NMF by alternative non-negative least squares using projected gradients
% Author: Chih-Jen Lin, National Taiwan University

% W,H: output solution
% Winit,Hinit: initial solution
% tol: tolerance for a relative stopping condition
% timelimit, maxiter: limit of time and iterations

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
projnorm = sqrt( norm(gradW(gradW<0 | W>0))^2 + norm( gradH(gradH<0 | H>0))^2 );
for iter=1:maxiter,
    % stopping condition
    if toc > timelimit | projnorm < tol, %Yue changed the stopping criterion
        break;
    end
    
    [W,gradW,iterW] = nlssubprob(V',H',W',tolW,1000); W = W'; gradW = gradW';
    if iterW==1,
        tolW = 0.1 * tolW;
    end
    
    [H,gradH,iterH] = nlssubprob(V,W,H,tolH,1000);
    if iterH==1,
        tolH = 0.1 * tolH;
    end
    
    temp = toc;
    objstr = [objstr;.5*norm(W*H - V,'fro')^2];
    etime = etime + toc - temp;
    timeaxis = [timeaxis, toc - etime];
    gradW = W*(H*H') - V*H';
    projnorm = sqrt( norm(gradW(gradW<0 | W>0))^2 + norm( gradH(gradH<0 | H>0))^2 );
end
time = toc - etime;
fprintf('\nIter = %d Final proj-grad norm %f\n', iter, projnorm);
fprintf('Running time %fs\n', time);

function [H,grad,iter] = nlssubprob(V,W,Hinit,tol,maxiter)

% H, grad: output solution and gradient
% iter: #iterations used
% V, W: constant matrices
% Hinit: initial solution
% tol: stopping tolerance
% maxiter: limit of iterations

H = Hinit; WtV = W'*V; WtW = W'*W;

alpha = 1; beta = 0.1;
for iter=1:maxiter,
    grad = WtW*H - WtV;
    projgrad = norm(grad(grad < 0 | H >0));
    if projgrad < tol,
        break
    end
    
    % search step size
    for inner_iter=1:20,
        Hn = max(H - alpha*grad, 0); d = Hn-H;
        gradd=sum(sum(grad.*d)); dQd = sum(sum((WtW*d).*d));
        suff_decr = 0.99*gradd + 0.5*dQd < 0;
        if inner_iter==1,
            decr_alpha = ~suff_decr; Hp = H;
        end
        if decr_alpha,
            if suff_decr,
                H = Hn; break;
            else
                alpha = alpha * beta;
            end
        else
            if ~suff_decr | isequal(Hp,Hn),
                H = Hp; break;
            else
                alpha = alpha/beta; Hp = Hn;
            end
        end
    end
end

if iter==maxiter,
    grad = WtW*H - WtV; % Yue added this line.
    fprintf('Max iter in nlssubprob\n');
end
