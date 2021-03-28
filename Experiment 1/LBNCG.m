%Log-barrier Newton-CG for NMF
function [W,H,iter_act,time,error,res,projnorm] = LBNCG(V,Winit,Hinit,tol,timelimit,maxiter)
W = Winit; H = Hinit; tic;
epsh = sqrt(tol);
mu = tol/4;
beta = .5;
eta = .5;
[m,r] = size(W); [~,n] = size(H);
countNC = 0;
countSOL = 0;
countMEO = 0;
count = 0;
etime = 0;
E = W*H - V; objnow = .5*norm(E,'fro')^2 - mu*sum(sum(log([W',H])));
gradW = (E*H')'; gradH = W'*E; grad = [gradW,gradH];
temp = toc;
pick1 = (W' <= epsh); pick2 = (H <= epsh);
gradtild = [gradW(pick1); gradH(pick2)];
gradbar = [gradW(~pick1); gradH(~pick2)];
Wt = W';
if isempty(gradtild) == 1
    res1 = 0;
else
    res1 = -min(min(gradtild),0);
end
res = max( norm( [gradbar;[Wt(pick1);H(pick2)].*gradtild] ), res1 );
projnorm = norm([gradW(gradW<0 | W'>0); gradH(gradH<0 | H>0)]);
fprintf('Initial residual %f\n', res);
fprintf('Initial projnorm %f\n', projnorm);
fprintf('Initial objective value %f\n', objnow);
fprintf('Initial error %f\n', .5*norm(E,'fro')^2);
etime = etime + toc - temp;
iter_sub = 0;
% c_mu = .5*mu;
c_mu = .1;
% xi_r = .5;
zhat = .1;
for iter = 1:maxiter
    if toc > timelimit
        break;
    end
    X_bar = min([W',H],1);
    if min(min(grad)) <= -tol || max(max(abs(X_bar.*grad))) > tol
        g = reshape(X_bar.*grad - mu*(X_bar./[W',H]),[r*(m+n),1]);
        [d,Hd,dtype,count] = CCG(g,epsh,W,H,E,count,zhat,c_mu,mu);
        S =  1./(max([W',H],1)); S = reshape(S,[r*(m+n),1]); s = norm(S.*d,'Inf');
        if dtype == -1
            countNC = countNC + 1;
            sgn = sign(g'*d); sgn = 1 - sgn^2 + sgn;
            d = -sgn*min( abs(d'*(Hd-2*epsh*d))/(norm(d)^3), beta/s )*d;
        else
            countSOL = countSOL + 1;
            d = min(1,beta/s)*d;
        end
        dM = reshape(d,[r,m+n]); dW = dM(:,1:m)'; dH = dM(:,m+1:m+n);
        %linesearch
        dnorm3 = norm(d)^3;
        alpha = 1;find = 0;
        for i = 1:50
            Wn = W + alpha*(min(W,1).*dW); Hn = H + alpha*(min(H,1).*dH); 
            En = Wn*Hn - V; objnew = .5*norm(En,'fro')^2 - mu*sum(sum(log([Wn',Hn])));
            if objnew - objnow < - eta/6*alpha^3*dnorm3
                W = Wn; H = Hn; E = En; objnow = objnew; find = 1;
                break;
            else
                alpha = alpha/2;
            end
        end
        if find == 0
            fprintf('Small stepsize in NCG line search\n');
            if dtype == 1
                zhat = zhat * .1; c_mu = c_mu * .1;
            end
            iter_sub = iter_sub + 1;
            if dtype == -1
                break;
            end
        end
    else
        break;
    end
    gradW = (E*H')'; gradH = W'*E; grad = [gradW,gradH];
end
time = toc-etime;
projnorm = norm([gradW(gradW<0 | W'>0); gradH(gradH<0 | H>0)]);
pick1 = (W' <= epsh); pick2 = (H <= epsh);
gradtild = [gradW(pick1); gradH(pick2)];
gradbar = [gradW(~pick1); gradH(~pick2)];
Wt = W';
if isempty(gradtild) == 1
    res1 = 0;
else
    res1 = -min(min(gradtild),0);
end
res = max( norm( [gradbar;[Wt(pick1);H(pick2)].*gradtild] ), res1 );
iter_act = iter-iter_sub;
error = .5*norm(E,'fro')^2;
fprintf('Ratio of NC, SOL and MEOstep is %.4f, %.4f and %.4f\n',countNC/iter, countSOL/iter,countMEO/iter)
fprintf('Iter = %d Final residual %f\n', iter_act, res);
fprintf('Final projnorm %f\n', projnorm);
fprintf('Final objective value %f\n', objnow);
fprintf('Final error %f\n', error);
fprintf('Running time %fs\n', time);

function [d,Hd,dtype,count] = CCG(g,eps,W,H,E,count,zhat,c_mu,mu)
[m,r_dim] = size(W);[~,n] = size(H);
WtW = W'*W; HHt = H*H';
M = sqrt(2*min([m,n,r_dim]))*(norm(E)+norm(W)*norm(H)+max(norm(WtW),norm(HHt))) + mu;
kap = (M+2*eps)/eps; 
% zeta = .5;
zhat = max(zhat,.5/3/kap);
% zhat = .1;%xi_r/3/kap;
c_mu = max(c_mu,.5*mu);
tau = sqrt(kap)/(sqrt(kap)+1); T = 4*kap^4/(1-sqrt(tau))^2;
dim = length(g); y = zeros(dim,1); r = g; p = -g; j=0; normg = norm(g);
Hp = Hessvec(p,eps,W,H,WtW,HHt,E,mu); count = count + 1;
if p'* Hp < eps * (p'*p)
    d = p;Hd=Hp;dtype = -1;
    return;
end
flag = 0;
Hy = zeros(dim,1);
while 1
    alpha = r'*r/(p'*Hp);
    y = y + alpha*p;    Hy = Hy + alpha*Hp;
    rn = r+alpha*Hp;
    beta = rn'*rn/(r'*r);
    p = -rn+beta*p;
    r = rn;
    j=j+1;
    Hp = Hessvec(p,eps,W,H,WtW,HHt,E,mu);
    count = count + 1;
    if flag == 0
        if y'*Hy < eps*(y'*y)
            d = y;Hd=Hy;dtype = -1;
            return;
        elseif norm(r) <= zhat*normg && norm(r,'inf') <= c_mu
            d = y;Hd = 0;dtype = 1;
            return;
        elseif p'*Hp < eps*(p'*p)
            d = p;Hd=Hp;dtype = -1;
            return;
        elseif norm(r) > sqrt(T)*tau^(j/2)*normg
            flag = 1;
            alpha = r'*r/(p'*Hp);
            yn = y + alpha*p; Hyn = Hy + alpha*Hp;
            y = zeros(dim,1);
            d = yn - y;
            Hd = Hyn;
            if d'*Hd < eps*(d'*d)
                dtype = -1;
                return;
            else
                r = g; p = -g; j = 0; Hp = Hessvec(p,eps,W,H,WtW,HHt,E,mu); count = count + 1; Hy = zeros(dim,1);
            end
        end
    else
        d = yn - y;
        Hd = Hyn - Hy;
        if d'*Hd < eps*(d'*d)
            dtype = -1;
            return;
        end
    end
end

function Hp = Hessvec(p,eps,W,H,WtW,HHt,E,mu)
[m,r_dim] = size(W);[~,n] = size(H);
dM = reshape(p,[r_dim,m+n]);
W_bar = min(W,1); H_bar = min(H,1);
dW = dM(:,1:m)'; dH = dM(:,m+1:m+n); XdW = W_bar.*dW; XdH = H_bar.*dH;
Hp = [ W_bar.*( E*XdH' + W*(XdH*H') + XdW*HHt ) ; H_bar'.*( E'*XdW + H'*(XdW'*W) + XdH'*WtW ) ]' + mu * ( ([W_bar',H_bar]./[W',H]).^2 ).*dM;
Hp = reshape(Hp,[r_dim*(m+n),1]) + 2*eps*p;

% function [cert,dW,dH,lambda,count] = MEO(W,H,E,epsh,SW,SH,count)
% [m,r_dim] = size(W); [~,n] = size(H); dim = (m+n)*r_dim;
% b = randn(r_dim,m+n); b = b/norm(b,'fro');
% r = -b; p = -r; j = 0; %y = zeros(r_dim,m+n);
% WtW = W'*W; HHt = H*H'; M = sqrt(2*min([m,n,r_dim]))*(norm(E)+norm(W)*norm(H)+max(norm(WtW),norm(HHt)));
% SdW = SW.*(p(:,1:m)'); SdH = SH.*p(:,m+1:m+n); Hp = [ SW.*(E*SdH' + W*(SdH*H') + SdW*HHt); SH'.*(E'*SdW + H'*(SdW'*W) + SdH'*WtW) ]' + (epsh/2)*p;
% pHp = sum(sum(p.*Hp)); r_square = sum(sum(r.*r));
% J = min(dim,1+ceil(.5*log(2.75*dim/.01^2)*sqrt(M/epsh)));
% while pHp > 0 && (r_square ~= 0)
%     alpha = r_square/pHp;
%     %y = y + alpha*p;
%     rn = r + alpha*Hp;
%     rn_square = sum(sum(rn.*rn));
%     beta = rn_square/r_square;
%     p = -rn + beta*p;
%     j=j+1;
%     r = rn;r_square = rn_square;
%     SdW = SW.*(p(:,1:m)'); SdH = SH.*p(:,m+1:m+n); Hp = [ SW.*(E*SdH' + W*(SdH*H') + SdW*HHt); SH'.*(E'*SdW + H'*(SdW'*W) + SdH'*WtW) ]' + (epsh/2)*p;
%     pHp = sum(sum(p.*Hp));
%     if j == J
%         break;
%     end
% end
% count = count + j + 1;
% p_square = sum(sum(p.*p)); lambda = pHp/p_square - epsh/2; dW = p(:,1:m)'/sqrt(p_square); dH = p(:,m+1:m+n)/sqrt(p_square);
% if pHp <= 0
%     cert = 0;
% else
%     cert = 1;
% end


