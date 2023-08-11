clc;
clear;

n = 50000;
m = 25000;
K = 1;
sigma = 0;

load data_sim_train.mat
Fea = data;
load data_sim_test.mat
fea = data;


load label_train.mat
Gnd = data;

load label_test.mat
gnd = data;

[num_0,Dim] = size(Fea);
[num_1,Dim] = size(fea);



for i = 1:num_0
    Fea(i,:) = Fea(i,:)/norm(Fea(i,:));
end

for i = 1:num_1
    fea(i,:) = fea(i,:)/norm(fea(i,:));
end

index = zeros(10,5000);
count = zeros(10,1);
for i=1:50000
    for j=1:10
        if j ~= 10
            if Gnd(i) == j
                count(j) = count(j)+1;
                index(j,count(j)) = i;
            end
        else
            if Gnd(i) == 0
                count(j) = count(j)+1;
                index(j,count(j)) = i;
            end
        end
    end
end



Label = zeros(num_0,10);
label = zeros(num_1,10);
for i =1:num_0
    if Gnd(i) == 0
        Label(i,10) =1;
    else
        Label(i,Gnd(i)) = 1;
    end
end

for i = 1:num_1
    if gnd(i) == 0
        label(i,10) =1;
    else
        label(i,gnd(i)) = 1;
    end
end



d = Dim;
W = eye(Dim);

% d = 500;
% W = normrnd(0,1,Dim,d)/sqrt(d);
% load matrix_mask.mat
% % 


B = normrnd(0,sigma,m,d);

%% transform 
Fea = Fea*W;
fea = fea*W;

n_0 = n/10;
Y = zeros(n, 10);

for j=1:10
    s_index = randperm(count(j));
    for l = 1:n_0
        X((j-1)*n_0+l,:) = Fea(index(j,s_index(l)),:);
        Y((j-1)*n_0+l,:) = Label(index(j,s_index(l)),:);
    end
end


M = zeros(m,n);

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






m_0 = m/100;
item = 1;
MX = zeros(m, d);
MY = zeros(m, 10);
% for j = 1:10
%     for l = 1:10
%          for z = 1: m_0 
%              for k = 1:K
%                  s_index_1 = ceil(rand*n_0);
%                  s_index_2 = ceil(rand*n_0);
%                  MX (item,:) = MX (item,:) + (X((j-1)*n_0+s_index_1,:)+ X((l-1)*n_0+s_index_2,:))/(2*K);
%                  MY (item,:) = MY (item,:) + (Y((j-1)*n_0+s_index_1,:)+ Y((l-1)*n_0+s_index_2,:))/(2*K);
%              end              
% %              MX (item,:) = MX (item,:)/norm(MX (item,:));
%              item = item+1;
%          end
%     end
% end     

MX = M*X; 
MY = M*Y;
% 
% MX= X;
% MY = Y;
MX = MX + B;



for i = 1:m
    MX(i,:) = MX(i,:)/norm(MX(i,:));
end

% M = zeros (m,n);
% for i = 1:m
%     r_index = randperm(n);
%     for j = 1:K
%         M(i,r_index(j)) = 1/K;
%     end
% end
% X = zeros(n,d);
% r_index = randperm(num_0);
% for i = 1:n
%     X(i,:) = Fea(r_index(i),:);
%     Y(i,:) = Label(r_index(i),:);
% end
% MX = M*X;
% MY = M*Y;

%% Performance Test 

ave = 1;

 
 
for it=1:1:ave

    
    YRFBB = zeros(m,10);

    for i=1:m  
       XAS(:,:,1,i) = MX(i,:);
       YRFBB(i,:) = MY(i,:);
    end
    for i =1:num_0
        XAS_test(:,:,1,i) = Fea(i,:); 
    end
    for i=1:10000
       X_test(:,:,1,i) = fea(i,:);
    end
    
    
    %%
    %%%% CNN RF BB for regression type for relay
    dataFRFChainSelection = XAS;
    labelsRFChainSelection = YRFBB;
    sizeInputFRFChainSelection = size(dataFRFChainSelection);
    sizeOutputFRFChainSelection = size(labelsRFChainSelection);
   
    valDataRFChainSelection = X_test;
    valLabelsFRFChainSelection = label;

%     YRFBB = YRFBB';
    %% DNN for HB. for relay
    %%%% settings.
    layersFRFChainSelection = [imageInputLayer(sizeInputFRFChainSelection(1:3),'Normalization', 'zerocenter');
        
%         fullyConnectedLayer(1024);
%         batchNormalizationLayer
%         reluLayer();
%         fullyConnectedLayer(512);
%         batchNormalizationLayer
%         reluLayer();
%         fullyConnectedLayer(256);
%         batchNormalizationLayer
%         reluLayer();
%         fullyConnectedLayer(128);
%         batchNormalizationLayer
%         reluLayer();
%         fullyConnectedLayer(64);
%         batchNormalizationLayer
%         reluLayer();
%         fullyConnectedLayer(32);
%         batchNormalizationLayer
%         reluLayer();
        %%%%%%%%%%%%%%
        fullyConnectedLayer(10)
        regressionLayer()
%           softmaxLayer
%         classificationLayer()
        ];
    
    miniBatchSize = 200;
    learnRate = 0.02*miniBatchSize/128;
    valFrequency = 500;
    max_fail = 100;
    epochs = 500;
    net.trainParam.goal = epochs; 
    
    optsFRFSelection = trainingOptions('sgdm', ...
    'InitialLearnRate',learnRate, ...
    'MaxEpochs', 250, ...
    'MiniBatchSize',miniBatchSize, ...
    'VerboseFrequency',valFrequency, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'Verbose',false, ...
    'ValidationData',{valDataRFChainSelection,valLabelsFRFChainSelection}, ...
    'ValidationFrequency',valFrequency, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.8, ...
    'ValidationPatience', Inf,...
    'LearnRateDropPeriod',20);
    convnetFRFSelection = trainNetwork(dataFRFChainSelection, labelsRFChainSelection, layersFRFChainSelection, optsFRFSelection);



    
  
    
%     for i=1:1:100*C_num
%         %[~,Out_pre(i)] = max(double(predict(convnetFRFSelection,reshape(fea(num-i,:),dim_l,dim_l) * B )));
%         [~,Out_pre(i)] = max(double(predict(convnetFRFSelection,fea(num-i,:))));% estimate RF precoder
%     end

%     Out_pre=[];
count=0;
Test_num=5000;

for i=1:1:Test_num
        i
        %% mix inference 
%     item_pre = zeros(1,10);   
%     for j = 1: 50
%         item_test = X_test(:,:,:,i);
%         a = ceil(rand*num_0);
%         
%         item_test = item_test + XAS_test(:,:,:,a);
%         item_test = item_test/2;
%         item_pre =  item_pre + double(predict(convnetFRFSelection,X_test(:,:,:,i)))- Label(a,:)/2;
%     end
%     [~,Out_pre] = max(item_pre);    
     [~,Out_pre] = max(double(predict(convnetFRFSelection,X_test(:,:,:,i)))); 
    if Out_pre == 10
        Out_pre = 0;
    end
    if Out_pre == gnd(i)
        count= count+1;
    end
end


count = count/Test_num


end



