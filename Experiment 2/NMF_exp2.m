%% Data generation
m = 300; n = 200; r = 10;
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
%% Initial matrices
Winit = abs(randn(m,r)); Hinit = abs(randn(r,n));
Winit = Winit/mean(Winit,'all'); Hinit = Hinit/mean(Hinit,'all');
% Winit = abs(randn(m,1)); Hinit = abs(randn(1,n));
% Winit = Winit/mean(Winit,'all'); Hinit = Hinit/mean(Hinit,'all');
% [W0,H0,iter_alspgrad,time_alspgrad,x_alspgrad,y_alspgrad] = alspgrad(V,Winit,Hinit,1e-8,10,1000);
% Winit = W0*ones(1,r)/5; Hinit = ones(r,1)*H0/2;
%% Methods
[W1,H1,iter_PNCG,time_PNCG,x_PNCG,y_PNCG] = PNCG(V,Winit,Hinit,1e-6,100,5000);
%%
[W2,H2,iter_alspgrad,time_alspgrad,x_alspgrad,y_alspgrad] = alspgrad(V,Winit,Hinit,1e-4,100,1000);
%%
[W3,H3,iter_pnm,time_pnm,x_pnm,y_pnm] = pnm_nmf(V,Winit,Hinit,1e-4,100,1000);
%% Plot
subplot(2,2,1)
semilogy(x_PNCG,y_PNCG,'Marker','o','MarkerIndices',[1:20:length(x_PNCG),length(x_PNCG)]);
hold on
semilogy(x_alspgrad,y_alspgrad,'Marker','*','MarkerIndices',[1:3:length(x_alspgrad),length(x_alspgrad)]);
semilogy(x_pnm,y_pnm,'Marker','diamond','MarkerIndices',[1:3:length(x_pnm),length(x_pnm)]);
hold off
lgd = legend('PNCG','alspgrad','pnm');
xlabel('Time(s)')
ylabel('Function Value')
title(['r = ' num2str(r)]);