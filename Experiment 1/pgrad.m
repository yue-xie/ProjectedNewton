function [W,H,iter_act,time,objnow,res,projnorm] = pgrad(V,Winit,Hinit,tol,timelimit,maxiter,epsh)

% NMF by projected gradients

% W,H: output solution
% Winit,Hinit: initial solution
% tol: tolerance for a relative stopping condition
% timelimit, maxiter: limit of time and iterations

W = Winit; H = Hinit; tic; etime = 0; iter_sub = 0;
E = W*H - V;
gradW = E*H'; gradH = W'*E;
objnow = .5*norm(E,'fro')^2;
find = 1;
projnorm = norm([gradW(gradW<0 | W>0); gradH(gradH<0 | H>0)]);
for iter=1:maxiter,
    % stopping condition
    if toc > timelimit | projnorm <= tol
        break;
    end
    if find == 1
        jmax = 50; find = 0;
        for j = 1:jmax
            theta = .5^(j-1);
            Wn = max(W - theta*gradW,0); Hn = max(H - theta*gradH,0);
            En = Wn*Hn - V; objnew = .5*norm(En,'fro')^2;
            if objnew - objnow < .5*sum(sum([Wn' - W',Hn - H].*[gradW',gradH]))
                W = Wn; H = Hn; E = En; objnow = objnew; find = 1;
                break;
            end
        end
    else
        jmax = 50; find = 0;
        for j = 1:jmax
            theta = .5^(j-1);
            Wn = max(W - theta*gradW,0); Hn = max(H - theta*gradH,0); dW = Wn - W; dH = Hn - H;
            dWH = W*dH + dW*H + dW*dH;
            if sum(sum(E.*dWH)) + .5*sum(sum(dWH.*dWH)) < .5*sum(sum([dW',dH].*[gradW',gradH]))
                W = Wn; H = Hn; E = E + dWH; find = 1; objnow = .5*norm(E,'fro')^2;
                break;
            end
        end
    end
    if find == 0
        temp = toc;
        fprintf('Stepsize too small in GD line search\n');
        etime = etime + toc - temp;
        iter_sub = iter_sub + 1;
    end
    gradW = E*H'; gradH = W'*E;
    projnorm = norm([gradW(gradW<0 | W>0); gradH(gradH<0 | H>0)]);
end
time = toc-etime;
pick1 = (W <= epsh); pick2 = (H <= epsh);
gradtild = [gradW(pick1); gradH(pick2)];
gradbar = [gradW(~pick1); gradH(~pick2)];
if isempty(gradtild) == 1
    res1 = 0;
else
    res1 = -min(min(gradtild),0);
end
res = max( norm( [gradbar;[W(pick1);H(pick2)].*gradtild] ), res1 );
iter_act = iter - iter_sub;
fprintf('\nIter = %d Final residual %f\n', iter_act, res);
fprintf('Final projnorm %f\n', projnorm);
fprintf('Final objective value %f\n', objnow);
fprintf('Running time %fs\n', time);
