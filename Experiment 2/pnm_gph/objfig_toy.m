% An example: plot the objective value vs. time using pnm_nmf function.
% Written by Pinghua Gong, Tsinghua University

clear
clc

% parameters
tol = 1e-13;
maxiter = 1e13;
times = 0.1; timee = 51.2;
obj_pnm = zeros(10,1); time_pnm = obj_pnm;

% generate random data
n = 2000; m = 1200; r = 10;
V = abs(randn(n,m));
Winit = abs(randn(n,r));
Hinit = abs(randn(r,m));

% objective values 
i=1; time=times;
while time <= timee,
  [W_pnm,H_pnm] = pnm_nmf(V,Winit,Hinit,tol,time,maxiter);
  obj_pnm(i) = 0.5*(norm(V-W_pnm*H_pnm,'fro')^2);    
  time_pnm(i)=time;
  i = i + 1;
  time = time * 2;
end

% plot
semilogx(time_pnm, obj_pnm, 'o-')
legend('pnm');
set(findobj('Type', 'line'), 'LineWidth', 1)  
xlabel('Time in seconds (logged scale)'); 
ylabel('Objective value');


