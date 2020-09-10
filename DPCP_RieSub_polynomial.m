%Riemannian subgradient methods with polynomially diminishing stepsize
close all; clear;


%% setup the data
D = 100; %ambient dimension
N = 1500; % number of inliers
ratio = 1 ./ (1 ./ 0.7 - 1); % outlier ratio
M = floor(N * ratio); % number of outliers
d = 0.9*D; % subspace dimension
c = D -d;
X = [normc( randn(d,N) );zeros(D-d,N)];
O = normc(randn(D,M));
Ytilde = [X O];
obj = @(B) sum(sqrt(sum((B'*Ytilde).^2,1)));
S_p = [zeros(d,c); eye(c,c)]; %target solution
% initialization
% [Bo,~] = eigs(Ytilde*Ytilde',c,'SM');
Bo = randn(D,c);



%full subgradient
mu_0 = .1;
Niter = 300;
B = Bo;
for i = 1:Niter
    
    [u,s,w] = svd(S_p'*B);
    dist1(i) = norm(B - S_p*u*w');
    
    mu = mu_0/sqrt(i);
    BY = B'*Ytilde;
    temp = sqrt(sum((BY).^2,1));
    indx = (temp>0);
    grad = Ytilde(:,indx)./temp(indx) *(BY(:,indx))';
    gradB = grad'*B;
    grad = grad - 0.5*B*(gradB+ gradB');
    
    B_plus = B - mu*grad;
    B_power = B_plus'*B_plus;
    [U,Sigma,V] = svd(B_power);
    SIGMA =diag(Sigma);
    B = B_plus*(U*diag(sqrt(1./SIGMA))*V');
    
end




%incremental subgradient
mu_0 = .1;
Niter = 300;
B = Bo;
for i = 1:Niter
    
    [u,s,w] = svd(S_p'*B);
    dist2(i) = norm(B - S_p*u*w');
    
    mu = mu_0/sqrt(i);
    for j = 1:M+N
        Y = Ytilde(:,j);
        BY = B'*Y;
        temp = norm(BY);
        if temp == 0
            continue;
        else
            grad = Y*(BY'/temp);
            gradB = grad'*B;
            grad = grad - 0.5*B*(gradB+ gradB');
            
            
            B_plus = B - mu*grad;
            %polar retraction 
            % B_power = B_plus'*B_plus;
            %[U,Sigma] = eig(B_power);
            % SIGMA =diag(Sigma);
            %B = B_plus*(U*diag(sqrt(1./SIGMA))*U');
            
            %qr retraction 
            [B,~] = qr(B_plus,0);
         end
    end
    
end



%stochastic subgradient
mu_0 = .1;
Niter = 300;
B = Bo;
for i = 1:Niter
    
    [u,s,w] = svd(S_p'*B);
    dist3(i) = norm(B - S_p*u*w');
    
    mu = mu_0/sqrt(i);
    for j = 1:M+N
        index = randperm(M+N,1);
        Y = Ytilde(:,index);
        BY = B'*Y;
        temp = norm(BY);
        if temp == 0
            continue;
        else
            grad = Y*(BY'/temp);
            gradB = grad'*B;
            grad = grad - 0.5*B*(gradB+ gradB');
            
            
            B_plus = B - mu*grad;
            %polar retraction 
            % B_power = B_plus'*B_plus;
            %[U,Sigma] = eig(B_power);
            % SIGMA =diag(Sigma);
            %B = B_plus*(U*diag(sqrt(1./SIGMA))*U');
            
            %qr retraction 
            [B,~] = qr(B_plus,0);
         end
    end
    
end




%%
figure
loglog(dist1,'-','linewidth',2,'MarkerIndices', 1:30:length(dist1),'MarkerSize',8);
hold on
loglog(dist2,'r-*','linewidth',2,'MarkerIndices', 1:30:length(dist2),'MarkerSize',8);
loglog(dist3,'k-o','linewidth',2,'MarkerIndices', 1:30:length(dist3),'MarkerSize',8);
ylim([0,3])
xlim([0 300])
set(gca, ...
    'LineWidth' , 2                     , ...
    'FontSize'  , 20              , ...
    'FontName'  , 'Times New Roman'         );
legend('R-Full, $\gamma_k = 0.1/\sqrt{k} $','R-Incremental, $\gamma_k = 0.1/ \sqrt{k} $',...
    'R-Stochastic, $\gamma_k = 0.1/ \sqrt{k} $','FontSize',20,'Interpreter','LaTex')
xlabel('Iteration','FontSize',25,'FontName','Times New Roman');
ylabel('dist$({\mathbf X}_k,{\cal X})$','FontSize',25,'FontName','Times New Roman','Interpreter','LaTex');
set(gca,'YDir','normal')
set(gcf, 'Color', 'white');
% export_fig 'DPCP_polynomial.pdf' -nocrop
