function [W,H,iter,time,obj,exitflag,res,projnorm] = IP(V,Winit,Hinit,tol,timelimit,maxiter,epsh,opt)
tic
[m,r] = size(Winit); [~,n] = size(Hinit);
if opt == 1
    options = optimoptions('fmincon','MaxFunctionEvaluations',inf,'MaxIterations',maxiter,'OptimalityTolerance',tol,'SpecifyObjectiveGradient',true,'SubproblemAlgorithm','cg','HessianMultiplyFcn',@(x,lambda,dx)HessMultFcn(x,lambda,dx,V,m,n,r));
elseif opt == 2
    options = optimoptions('fmincon','MaxFunctionEvaluations',inf,'MaxIterations',maxiter,'OptimalityTolerance',tol,'SpecifyObjectiveGradient',true);
else
    options = optimoptions('fmincon','MaxFunctionEvaluations',inf,'MaxIterations',maxiter,'OptimalityTolerance',tol);
end
lb = zeros(r*(m+n),1);
ub = inf*ones(r*(m+n),1);
x0 = reshape([Winit',Hinit],(m+n)*r,1);
[x,obj,exitflag,output] = fmincon(@(x)myfun(x,V,m,n,r),x0,[],[],[],[],lb,ub,[],options);
time = toc;
iter = output.iterations;
X = reshape(x,r,m+n);
W = X(:,1:m)';
H = X(:,m+1:m+n);
E = W*H - V;
gradW = E*H';
gradH = W'*E;
projnorm = norm([gradW(W > 0 | gradW < 0);gradH(H > 0 | gradH < 0)]);
pick1 = (W <= epsh); pick2 = (H <= epsh);
gradtild = [gradW(pick1); gradH(pick2)];
gradbar = [gradW(~pick1); gradH(~pick2)];
if isempty(gradtild) == 1
    res1 = 0;
else
    res1 = -min(min(gradtild),0);
end
res = max( norm( [gradbar;[W(pick1);H(pick2)].*gradtild] ), res1 );
fprintf('\nIter = %d Final residual %f\n', iter, res);
fprintf('Final projnorm %f\n', projnorm);
fprintf('Final objective value %f\n', obj);
fprintf('Running time %fs\n', time);

function [f,g] = myfun(x,V,m,n,r)
X = reshape(x,r,m+n);
W = X(:,1:m)'; H = X(:,m+1:m+n);
E = W*H - V;
f = .5*norm(E,'fro')^2;
if nargout > 1
    G = [H*E',W'*E];
    g = reshape(G,(m+n)*r,1);
end

function Hdx = HessMultFcn(x,lambda,dx,V,m,n,r)
X = reshape(x,[r,m+n]);
W = X(:,1:m)'; H = X(:,m+1:m+n);
dX = reshape(dx,[r,m+n]);
dW = dX(:,1:m)'; dH = dX(:,m+1:m+n);
E = W*H - V;
HdX = [ E*dH' + W*(dH*H') + dW*(H*H') ; E'*dW + H'*(dW'*W) + dH'*(W'*W) ]';
Hdx = reshape(HdX,(m+n)*r,1);
