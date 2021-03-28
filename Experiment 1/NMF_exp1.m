%% Data generation
m = 150; n = 100; r = 15;
W0 = zeros(m,r); H0 = zeros(r,n);
for i = 1:m
    for j = 1:r
        W0(i,j) = abs(randn)*(rand<=.4);
    end
end
for i = 1:r
    for j = 1:n
        H0(i,j) = abs(randn)*(rand<=.4);
    end
end
V = W0*H0;
V = V + randn(m,n)*(mean(V,'all')*.05);
avV = mean(abs(V),'all');
V = V/avV;
noise = .5*norm(V - W0*H0/avV,'fro')^2;
%%
rep = 5;
iter_PNCG = zeros(rep,1);time_PNCG = zeros(rep,1);obj_PNCG = zeros(rep,1);res_PNCG = zeros(rep,1);projnorm_PNCG = zeros(rep,1);
iter_pgrad = zeros(rep,1);time_pgrad = zeros(rep,1);obj_pgrad = zeros(rep,1);res_pgrad = zeros(rep,1);projnorm_pgrad = zeros(rep,1);
iter_IP = zeros(rep,1);time_IP = zeros(rep,1);obj_IP = zeros(rep,1);res_IP = zeros(rep,1);projnorm_IP = zeros(rep,1);
iter_LBNCG = zeros(rep,1);time_LBNCG = zeros(rep,1);obj_LBNCG = zeros(rep,1);res_LBNCG = zeros(rep,1);projnorm_LBNCG = zeros(rep,1);
for num = 1:rep
%% Initial matrices
Winit = abs(randn(m,r)); Hinit = abs(randn(r,n));
Winit = Winit/mean(Winit,'all'); Hinit = Hinit/mean(Hinit,'all');
%% Methods
[W1,H1,iter_PNCG(num),time_PNCG(num),obj_PNCG(num),res_PNCG(num),projnorm_PNCG(num)] = PNCG(V,Winit,Hinit,1e-6,100,5000);

[W2,H2,iter_pgrad(num),time_pgrad(num),obj_pgrad(num),res_pgrad(num),projnorm_pgrad(num)] = pgrad(V,Winit,Hinit,1e-4,100,5000,1e-3);

[W3,H3,iter_IP(num),time_IP(num),obj_IP(num),exitflag,res_IP(num),projnorm_IP(num)] = IP(V,Winit,Hinit,1e-4,100,50,1e-3,1);

[W4,H4,iter_LBNCG(num),time_LBNCG(num),obj_LBNCG(num),res_LBNCG(num),projnorm_LBNCG(num)] = LBNCG(V,Winit,Hinit,1e-4,60,10000);
end
row1 = [mean(iter_PNCG),mean(time_PNCG),mean(obj_PNCG),mean(res_PNCG),mean(projnorm_PNCG)];
row2 = [mean(iter_pgrad),mean(time_pgrad),mean(obj_pgrad),mean(res_pgrad),mean(projnorm_pgrad)];
row3 = [mean(iter_IP),mean(time_IP),mean(obj_IP),mean(res_IP),mean(projnorm_IP)];
row4 = [mean(iter_LBNCG),mean(time_LBNCG),mean(obj_LBNCG),mean(res_LBNCG),mean(projnorm_LBNCG)];
%%
tab = [];
tab = [tab;row1;row2;row3;row4];
