%Projected Newton-CG for NMF
function [W,H,iter,time,timeaxis,objstr] = PNCG(V,Winit,Hinit,tol,timelimit,maxiter)
W = Winit; H = Hinit; tic;
epsh = sqrt(tol);
[m,r] = size(W); [~,n] = size(H);
objstr = [];
countNC = 0;
countGD = 0;
countMEO = 0;
count = 0;
timeaxis = [];
etime = 0;
E = W*H - V; objnow = .5*norm(E,'fro')^2;
gradW = (E*H')'; gradH = W'*E;
pick1 = (W' <= epsh); pick2 = (H <= epsh);
gradtild = [gradW(pick1); gradH(pick2)];
gradbar = [gradW(~pick1); gradH(~pick2)];
if isempty(gradtild) == 1
    res1 = 0; res2 = 0;
    res3 = norm(gradbar);
else
    res1 = -min(min(gradtild),0); Wt = W';
    res2 = norm([Wt(pick1);H(pick2)].*gradtild);
    if isempty(gradbar) == 1
        res3 = 0;
    else
        res3 = norm(gradbar);
    end
end
zhat = .1;
for iter = 1:maxiter
    if toc > timelimit
        break;
    end
    if (res1 > epsh^1.5) || (res2 > epsh^2)
        countGD = countGD + 1;
        count = count + 1;
        %linesearch
        jmax = 50; find = 0;
        for j = 1:jmax
            theta = .5^(j-1);
            Wn = max(W - theta*gradW',0); Hn = max(H - theta*gradH,0);
            En = Wn*Hn - V; objnew = .5*norm(En,'fro')^2;
            if objnew - objnow < .5*sum(sum([Wn'-W',Hn-H].*[gradW,gradH]))
                W = Wn; H = Hn; E = En; find = 1;
                objnow = objnew;
                break;
            end
        end
        if find == 0
            temp = toc;
            fprintf('Small stepsize in GD line search\n');
            etime = etime + toc - temp;
        end
    elseif res3 > tol
        [d,Hd,dtype,count] = CCG(gradbar,epsh,W,H,E,count,zhat);
        if dtype == -1
            countNC = countNC + 1;
            sgn = sign(gradbar'*d); sgn = 1 - sgn^2 + sgn;
            d = -sgn*abs(d'*(Hd-2*epsh*d))*d/(norm(d)^3);
        end
        dM = zeros(r,m+n);
        dM([~pick1,~pick2]) = d; dW = dM(:,1:m)'; dH = dM(:,m+1:m+n);
        %linesearch
        dnorm2 = d'*d;
        alpha = 1;find = 0;
        for i = 1:50
            Wn = max(W + alpha*dW,0); Hn = max(H + alpha*dH,0); DW = Wn - W; DH = Hn - H;
            DWH = DW*H + W*DH + DW*DH;
            if sum(sum(E.*DWH)) + .5*sum(sum(DWH.*DWH)) < - .2*alpha^2*epsh*dnorm2
                W = Wn;H = Hn; E = E + DWH; objnow = .5*norm(E,'fro')^2; find = 1;
                break;
            else
                alpha = alpha/2;
            end
        end
        if find == 0
            temp = toc;
            fprintf('Small stepsize in NCG line search\n');
            etime = etime + toc - temp;
            if dtype == 1
                zhat = zhat * .1;
            end
        end
    else
        SW = W; SH = H; SW(~pick1') = 1; SH(~pick2) = 1;
        [cert,dW,dH,lambda,count] = MEO(W,H,E,epsh,SW,SH,count);
        if cert == 0
            sgn = sign( sum(sum(gradW'.*(SW.*dW))) + sum(sum(gradH.*(SH.*dH))) ); sgn = 1 - sgn^2 + sgn;
            dW = -sgn*abs(lambda)*dW; dH = -sgn*abs(lambda)*dH;
            alpha = 1;find = 0;
            for i = 1:50
                Wn = max(W + alpha*(SW.*dW),0); Hn = max(H + alpha*(SH.*dH),0); DW = Wn - W; DH = Hn - H;
                DWH = DW*H + W*DH + DW*DH;
                if sum(sum(E.*DWH)) + .5*sum(sum(DWH.*DWH)) < - .2*alpha^2*(abs(lambda))^3
                    W = Wn;H = Hn;E = E + DWH; objnow = .5*norm(E,'fro')^2; find = 1;
                    break;
                else
                    alpha = alpha/2;
                end
            end
            countMEO = countMEO + 1;
            if find == 0
                temp = toc;
                fprintf('Small stepsize in MEO line search\n');
                etime = etime + toc - temp;
            end
        else
            break;
        end
    end
    gradW = (E*H')'; gradH = W'*E;
    pick1 = (W' <= epsh); pick2 = (H <= epsh);
    gradtild = [gradW(pick1); gradH(pick2)];
    gradbar = [gradW(~pick1); gradH(~pick2)];
    if isempty(gradtild) == 1
        res1 = 0; res2 = 0;
        res3 = norm(gradbar);
    else
        res1 = -min(min(gradtild),0); Wt = W';
        res2 = norm([Wt(pick1);H(pick2)].*gradtild);
        if isempty(gradbar) == 1
            res3 = 0;
        else
            res3 = norm(gradbar);
        end
    end
    timeaxis = [timeaxis,toc-etime];
    objstr = [objstr,objnow];
end
time = toc-etime;
fprintf('\nRatio of NC, SOL, GDstep and MEOstep is %.4f, %.4f, %.4f and %.4f\n',countNC/iter, 1 - countNC/iter - countGD/iter - countMEO/iter,countGD/iter,countMEO/iter)

function [d,Hd,dtype,count] = CCG(g,eps,W,H,E,count,zhat)
[m,r_dim] = size(W);[~,n] = size(H);
WtW = W'*W; HHt = H*H';
M = sqrt(2*min([m,n,r_dim]))*(norm(E)+norm(W)*norm(H)+max(norm(WtW),norm(HHt)));
zeta = .5;
kap = (M+2*eps)/eps; zhat = max(zhat,zeta/3/kap);
tau = sqrt(kap)/(sqrt(kap)+1); T = 4*kap^4/(1-sqrt(tau))^2;
dim = length(g); y = zeros(dim,1); r = g; p = -g; j=0; normg = norm(g);
Hp = Hessvec(p,eps,W,H,WtW,HHt,E); count = count + 1;
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
    Hp = Hessvec(p,eps,W,H,WtW,HHt,E);
    count = count + 1;
    if flag == 0
        if y'*Hy < eps*(y'*y)
            d = y;Hd=Hy;dtype = -1;
            return;
        elseif norm(r) <= zhat*normg
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
                r = g; p = -g; j = 0; Hp = Hessvec(p,eps,W,H,WtW,HHt,E); count = count + 1; Hy = zeros(dim,1);
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

function Hp = Hessvec(p,eps,W,H,WtW,HHt,E)
[m,r_dim] = size(W);[~,n] = size(H);
dM = zeros(r_dim,m+n);
dM([W',H] > eps) = p; dW = dM(:,1:m)'; dH = dM(:,m+1:m+n);
Hp = [ E*dH' + W*(dH*H') + dW*HHt ; E'*dW + H'*(dW'*W) + dH'*WtW ]';
Hp = Hp([W',H] > eps) + 2*eps*p;

function [cert,dW,dH,lambda,count] = MEO(W,H,E,epsh,SW,SH,count)
[m,r_dim] = size(W); [~,n] = size(H); dim = (m+n)*r_dim;
b = randn(r_dim,m+n); b = b/norm(b,'fro');
r = -b; p = -r; j = 0; %y = zeros(r_dim,m+n);
WtW = W'*W; HHt = H*H'; M = sqrt(2*min([m,n,r_dim]))*(norm(E)+norm(W)*norm(H)+max(norm(WtW),norm(HHt)));
SdW = SW.*(p(:,1:m)'); SdH = SH.*p(:,m+1:m+n); Hp = [ SW.*(E*SdH' + W*(SdH*H') + SdW*HHt); SH'.*(E'*SdW + H'*(SdW'*W) + SdH'*WtW) ]' + (epsh/2)*p;
pHp = sum(sum(p.*Hp)); r_square = sum(sum(r.*r));
J = min(dim,1+ceil(.5*log(2.75*dim/.01^2)*sqrt(M/epsh)));
while pHp > 0 && (r_square ~= 0)
    alpha = r_square/pHp;
    %y = y + alpha*p;
    rn = r + alpha*Hp;
    rn_square = sum(sum(rn.*rn));
    beta = rn_square/r_square;
    p = -rn + beta*p;
    j=j+1;
    r = rn;r_square = rn_square;
    SdW = SW.*(p(:,1:m)'); SdH = SH.*p(:,m+1:m+n); Hp = [ SW.*(E*SdH' + W*(SdH*H') + SdW*HHt); SH'.*(E'*SdW + H'*(SdW'*W) + SdH'*WtW) ]' + (epsh/2)*p;
    pHp = sum(sum(p.*Hp));
    if j == J
        break;
    end
end
count = count + j + 1;
p_square = sum(sum(p.*p)); lambda = pHp/p_square - epsh/2; dW = p(:,1:m)'/sqrt(p_square); dH = p(:,m+1:m+n)/sqrt(p_square);
if pHp <= 0
    cert = 0;
else
    cert = 1;
end


