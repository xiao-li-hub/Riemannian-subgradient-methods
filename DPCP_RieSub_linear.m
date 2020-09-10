%Riemannian subgradient methods with geometrically diminishing stepsize
close all; clear;


%% setup the data
D = 100; %ambient dimension
N = 15*D; % number of inliers
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
tic;
mu_0 = 1e-1;
beta = .75;
Niter = 100;
B = Bo;
for i = 1:Niter

    [u,s,w] = svd(S_p'*B);
    dist1(i) = norm(B - S_p*u*w');
    
    mu = mu_0*beta^(i);
    BY = B'*Ytilde;
    temp = sqrt(sum((BY).^2,1));
    indx = (temp>0);
    grad = Ytilde(:,indx)./temp(indx) *(BY(:,indx))';
    gradB = grad'*B;
    grad = grad - 0.5*B*(gradB+ gradB');
    
    dist_x2x_k(i) = norm(B-Bo,'fro');
    gradnorm(i) = norm(grad,'fro');
    
    B_plus = B - mu*grad;
    B_power = B_plus'*B_plus;
    [U,Sigma,V] = svd(B_power);
    SIGMA =diag(Sigma);
    B = B_plus*(U*diag(sqrt(1./SIGMA))*V');
    
     
    
end
toc;


%incremental subgradient
tic;
mu_0 = 1e-1;
beta = .5;
Niter = 100;
B = Bo;
for i = 1:Niter
    
    [u,s,w] = svd(S_p'*B);
    dist2(i) = norm(B - S_p*u*w');
    
    mu = mu_0*beta^(i);
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
            %grad = grad - 0.5*B*(grad'*B+ B'*grad);
            
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
toc;


%stochastic subgradient
tic;
mu_0 = 1e-1;
beta = .55;
Niter = 100;
B = Bo;
for i = 1:Niter
    [u,s,w] = svd(S_p'*B);
    dist3(i) = norm(B - S_p*u*w');
    
    mu = mu_0*beta^(i);
    for j = 1:M+N
        index = randperm(M+N,1);
        Y = Ytilde(:,index);
        BY = B'*Y;
        temp = norm(BY);
        %  temp = sqrt(sum((B'*Ytilde(:,index)).^2));
        if temp == 0
            continue;
        else
            grad = Y*(BY'/temp);
            gradB = grad'*B;
            grad = grad - 0.5*B*(gradB+ gradB');
            %  grad = grad - 0.5*B*(grad'*B+ B'*grad);
            
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
toc;


%%
figure
semilogy(dist1,'-','linewidth',2,'MarkerIndices', 1:10:length(dist1),'MarkerSize',8);
hold on
semilogy(dist2,'r-*','linewidth',2,'MarkerIndices', 1:10:length(dist2),'MarkerSize',8);
semilogy(dist3,'k-o','linewidth',2,'MarkerIndices', 1:10:length(dist3),'MarkerSize',8);
ylim([1e-10,max(dist1)*1.2])
set(gca, ...
    'LineWidth' , 2                     , ...
    'FontSize'  , 20              , ...
    'FontName'  , 'Times New Roman'         );
legend('R-Full, $\gamma_k = 0.1 \times 0.75^k$','R-Incremental, $\gamma_k = 0.1 \times 0.5^k$',...
    'R-Stochastic, $\gamma_k = 0.1 \times 0.55^k$',...
    'FontSize',20,'Interpreter','LaTex')
xlabel('Iteration','FontSize',25,'FontName','Times New Roman');
ylabel('dist$({\mathbf X}_k,{\cal X})$','FontSize',25,'FontName','Times New Roman','Interpreter','LaTex');
set(gca,'YDir','normal')
set(gcf, 'Color', 'white');