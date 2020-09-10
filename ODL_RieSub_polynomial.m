%Riemannian subgradient methods with polynomially diminishing stepsize
close all; clear;

%% setup the data
theta = .3;   % sparsity level
D = 30;   % dimension
p = 1.5;   % sample complexity (as power of n)
m = round(10*D^p);    % number of measurements
Q = randU(D);     % a uniformly random orthogonal matrix
X = randn(D, m).*(rand(D, m) <= theta);   % iid Bern-Gaussian model
Xtilde = Q*X;
% random initialization
Bo = orth(randn(D));

%% geometrical constant

%full subgradient
mu_0 = 1e-3;
Niter = 1000;
B = Bo;
for i = 1:Niter
    dist1(i) = sum( abs( max(abs(B'*Q),[],2) - ones(D,1) )  );
    
    mu = mu_0/sqrt(i);
    grad = Xtilde*sign(Xtilde'*B); %Euclidean subgradient
    gradB = grad'*B;
    grad = grad - 0.5*B*(gradB+ gradB'); %Riemannian subgradient
    
    B_plus = B - mu*grad;
    [B,~] = qr(B_plus,0);    %update and retraction
   
end



%incremental subgradient
mu_0 = 1e-3;
Niter = 1000;
B = Bo;
for i = 1:Niter
    dist2(i) = sum( abs( max(abs(B'*Q),[],2) - ones(D,1) )  );
        
    mu = mu_0/sqrt(i);
    for j = 1:m
        Y = Xtilde(:,j);
        
        grad = Y*sign(Y'*B);
        gradB = grad'*B;
        grad = grad - 0.5*B*(gradB+ gradB');
        
        B_plus = B - mu*grad;
        [B,~] = qr(B_plus,0);
        
    end
end





%stochastic subgradient
mu_0 = 1e-3;
Niter = 1000;
B = Bo;
for i = 1:Niter
    dist3(i) = sum( abs( max(abs(B'*Q),[],2) - ones(D,1) )  );
    
    mu = mu_0/sqrt(i);
    for j = 1:m
        index = randperm(m,1);
        Y = Xtilde(:,index);
        
        grad = Y*sign(Y'*B);
        gradB = grad'*B;
        grad = grad - 0.5*B*(gradB+ gradB');
        
        
        B_plus = B - mu*grad;
        [B,~] = qr(B_plus,0);
        
    end
    
end


%%
figure
loglog(dist1,'-','linewidth',2,'MarkerIndices', 1:30:length(dist1),'MarkerSize',8);
hold on
loglog(dist2,'r-*','linewidth',2,'MarkerIndices', 1:30:length(dist2),'MarkerSize',8);
loglog(dist3,'k-o','linewidth',2,'MarkerIndices', 1:30:length(dist3),'MarkerSize',8);
ylim([0,max(dist1)*1.05])
set(gca, ...
    'LineWidth' , 2                     , ...
    'FontSize'  , 20              , ...
    'FontName'  , 'Times New Roman'         );
legend('R-Full, $\gamma_k = 10^{-3} / \sqrt{k}$','R-Incremental, $\gamma_k = 10^{-3} / \sqrt{k}$',...
    'R-Stochastic, $\gamma_k = 10^{-3} / \sqrt{k}$',...
    'FontSize',20,'Interpreter','LaTex')
xlabel('Iteration','FontSize',25,'FontName','Times New Roman');
ylabel('$Er({\mathbf X}_k,{\mathbf A})$','FontSize',25,'FontName','Times New Roman','Interpreter','LaTex');
set(gca,'YDir','normal')
set(gcf, 'Color', 'white');

