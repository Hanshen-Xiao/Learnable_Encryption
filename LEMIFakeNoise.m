clc;
clear;

% load data_cifar100_train.mat
% Fea = data;
% load label_train.mat
% Label = data;

load data_sim_train.mat
Fea = data;
load data_sim_test.mat
fea = data;
gpuDevice(1)

load label_train.mat
Label = data;

load label_test.mat
label = data;   

index = zeros(10,5000);
count = zeros(10,1);

index_pub = zeros(10,1000);

for i=1:50000
    for j=1:10
        if j ~= 10
            if Label(i) == j
                count(j) = count(j)+1;
                index(j,count(j)) = i;
            end
        else
            if Label(i) == 0
                count(j) = count(j)+1;
                index(j,count(j)) = i;
            end
        end
    end
end

count = zeros(10,1);
for i=1:10000
    for j=1:10
        if j ~= 10
            if label(i) == j
                count(j) = count(j)+1;
                index_pub(j,count(j)) = i;
            end
        else
            if label(i) == 0
                count(j) = count(j)+1;
                index_pub(j,count(j)) = i;
            end
        end
    end
end


% Fea = normalize(Fea);
% fea = normalize(Fea);

for i=1:50000
    Fea(i,:) = Fea(i,:)/norm(Fea(i,:));
end

for i=1:10000
    fea(i,:) = fea(i,:)/norm(fea(i,:));
end

out_d = 500;


n = 2000;
m = 6000;
K = 10;
num = 100;
s = size(Fea);
d = s(2);
X = zeros(n,d);
n_0 = n/10;
c = 0.03; 
log_det_avg  = 0;
A_avg = (zeros(m,m));

%% Fake Noise
n_pub = 1000;
X_pub = zeros(n_pub,d);
n_pub_0 = n_pub/10;
for j = 1: 10
    for l = 1:n_pub_0 
        X_pub((j-1)*n_pub_0+l,:) = fea(index_pub(j,l),:);
    end
end

M_pub = zeros(m,n_pub);
item_M = 1;

for z = 1:10
     for l = 1:10
         for h = 1:m/100
             s_index_1 = randperm(n_pub_0);
             s_index_2 = randperm(n_pub_0);
             for x = 1:K
                 M_pub(item_M, (z-1)*n_pub_0+s_index_1(x)) = 1/K/2;
                 M_pub(item_M, (l-1)*n_pub_0+s_index_2(x)) = 1/K/2;
             end
                        item_M = item_M +1;
          end
     end
end
MX_pub = M_pub*X_pub/sqrt(out_d);
Sig_pub =   MX_pub*MX_pub';


% for i = 1:num
%     i
%     for j = 1: 10
%         s_index = randperm(count(j));
%         for l = 1:n_0 
%             X((j-1)*n_0+l,:) = Fea(index(j,s_index(l)),:);
%         end
%     end
%     MX = zeros(m,d);
%     %% 2-mix 
%     m_0 = m/100;
%     item = 1;
%     m_0 = m/100;
%     for j = 1:10
%         for l = 1:10
%             for z = 1: m_0 
%                 for k = 1:K
%                     s_index_1 = ceil(rand*n_0);
%                     s_index_2 = ceil(rand*n_0);
%                     MX (item,:) = MX (item,:) + (X((j-1)*n_0+s_index_1,:)+ X((l-1)*n_0+s_index_2,:))/(2*K);
%                 end              
%                 MX (item,:) = MX (item,:)/norm(MX (item,:));
%                 item = item+1;
%             end
%         end
%     end
%     
%     %% Type II
%     A = MX*MX';
%     e = ones(1,m);
%     
%     v_0 = e*A*e'/m;
%     v_1 = (trace(A)-v_0)/(m-1);
%     H = v_0*e'*e/m+ v_1*(eye(m)-e'*e/m);
%     A_avg = A_avg + H/(num);
%     
% %     A_avg = A_avg+A/num;
%     
%     s = gpuArray(eig(H));
%     s = s/c^2 + ones(m,1);
%     log_det_avg = log_det_avg + sum(log(s))/num;
%     
% end
% 
% s = gpuArray(eig(A_avg));
% s = s/c^2 + ones(m,1);
% log_avg_det = sum(log(s));
% log_avg_det - log_det_avg


%% Individual 
num_0 = 5;
num_ind = 10;

type_1 = zeros(1,num_ind*num*num_0);
m_0 = m/10;
count_item = 1;
X_0 = zeros(n,d);

type_1_ind = zeros(1, num_ind);


% M = zeros(m,n);
%            item_M = 1;
%          
% for z = 1:10
%      for l = 1:10
%          for h = 1:m/100
%              s_index_1 = randperm(n_0);
%              s_index_2 = randperm(n_0);
%              for x = 1:K
%                  M(item_M, (z-1)*n_0+s_index_1(x)) = 1/K/2;
%                  M(item_M, (l-1)*n_0+s_index_2(x)) = 1/K/2;
%              end
%                         item_M = item_M +1;
%           end
%      end
% end

for o = 1:num_ind
    for i = 1:num
        for j = 1: 10
            s_index = randperm(count(j));
            for l = 1:n_0 
                X((j-1)*n_0+l,:) = Fea(index(j,s_index(l)),:);
            end
        end
        X_i = X;
        X_i(1, :) = fea(index_pub(1,104+o), :);



        for j = 1:num_0
            M = zeros(m,n);
            %% single mix 
%             m_0 = m/10;
%             for z = 1:10
%                 for l = 1:m_0
%                         s_index= randperm(n_0);
%                         for x =1:K
%                             M((z-1)*m_0+l,(z-1)*n_0+s_index(x)) = 1/K;
%                         end
%                 end
%             end
            %%  2-mix
            item_M = 1;
            
            for z = 1:10
                for l = 1:10
                    for h = 1:m/100
                        s_index_1 = randperm(n_0);
                        s_index_2 = randperm(n_0);
                        for x = 1:K
                            M(item_M, (z-1)*n_0+s_index_1(x)) = 1/K/2;
                            M(item_M, (l-1)*n_0+s_index_2(x)) = 1/K/2;
                        end
                        item_M = item_M +1;
                    end
                end
            end
%             M= eye(n);
    %         m =n;
    %         MuX = M*X_0;
    
        %% 3-mix

%         item_M = 1;
%             
%             for z = 1:10
%                 for l = 1:10
%                     for g = 1:10
%                         for h = 1:m/1000
%                             s_index_1 = randperm(n_0);
%                             s_index_2 = randperm(n_0);
%                             s_index_3 = randperm(n_0);
%                         for x = 1:K
%                             M(item_M, (z-1)*n_0+s_index_1(x)) = 1/K/3;
%                             M(item_M, (l-1)*n_0+s_index_2(x)) = 1/K/3;
%                             M(item_M, (g-1)*n_0+s_index_3(x)) = 1/K/2;
%                         end
%                         item_M = item_M +1;
%                         end
%                     end
%                 end
%             end


            
            
            
            MuX = M*X/sqrt(out_d); 
            MuX_i = M*X_i/sqrt(out_d);
            
            
            
%             for u = 1:m
%                 MuX(u,:) = MuX(u,:)/norm(MuX(u,:));
%                 MuX_i(u,:) = MuX_i(u,:)/norm(MuX(u,:));
%             end

            Sigma_0 = gpuArray(MuX*MuX' + Sig_pub + c^2*eye(m));
            Sigma_1 = gpuArray(MuX_i*MuX_i' + Sig_pub + c^2*eye(m));
            inv_Sigma_0 = gpuArray(inv(Sigma_0));
            inv_Sigma_1 = gpuArray(inv(Sigma_1));
            type_1(count_item) = (trace(Sigma_0*inv_Sigma_1) + trace(Sigma_1*inv_Sigma_0)-2*m);
            count_item = count_item+1; 
        end
       

    end
   p = out_d/2*(n/50000)*sum(type_1(:,(o-1)*num_0*num+1:o*num_0*num))/(num_0*num)
   type_1_ind(o) = p;
end
sum(type_1)/length(type_1)


%% single mix 
% m_0 = m/10;
% for i = 1:num
%     item = 1;
%     for j = 1:10
%         for z = 1: m_0
%                 for k = 1:K
%                     s_index_1 = ceil(rand*n_0);                 
%                     MX (i,item,:) = MX (i,item,:) + (X(i,(j-1)*n_0+s_index_1,:))/(K);
%                 end         
%                 v = 0;
%                 for r = 1:d
%                     v = v+ MX (i,item,r)^2;
%                 end
%                 MX (i,item,:) = MX (i,item,:)/sqrt(v);
%                 item = item+1;
%         end
%     end
% end
            


