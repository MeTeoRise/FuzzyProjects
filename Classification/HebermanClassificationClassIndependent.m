close all;
clear all;

%% Data preparation

data = importdata('haberman.data');

[m,n] = size(data);

% data = data(randperm(size(data, 1)), :);

idx=randperm(length(data));
trnIdx=idx(1:round(length(idx)*0.6));
chkIdx=idx(round(length(idx)*0.6)+1:round(length(idx)*0.8));
tstIdx=idx(round(length(idx)*0.8)+1:end);
trnX=data(trnIdx,1:end-1);
chkX=data(chkIdx,1:end-1);
tstX=data(tstIdx,1:end-1);

xmin=min(trnX,[],1);
xmax=max(trnX,[],1);
trnX=(trnX-repmat(xmin,[length(trnX) 1]))./(repmat(xmax,[length(trnX) 1])-repmat(xmin,[length(trnX) 1]));
chkX=(chkX-repmat(xmin,[length(chkX) 1]))./(repmat(xmax,[length(chkX) 1])-repmat(xmin,[length(chkX) 1]));
tstX=(tstX-repmat(xmin,[length(tstX) 1]))./(repmat(xmax,[length(tstX) 1])-repmat(xmin,[length(tstX) 1]));

trainingData=[trnX data(trnIdx,end)];
validationData=[chkX data(chkIdx,end)];
chkData=[tstX data(tstIdx,end)];

%% Generate fis
radii1 = 0.35;
fis1 = genfis2(trainingData(:,1:end-1),trainingData(:,end),radii1);
for i = 1:size(fis1.output.mf, 2)        
    constt = fis1.output.mf(i).params(end);
    fis1.output.mf(i).type = 'constant';
    fis1.output.mf(i).params = constt;       
end

radii2 = 0.65;
fis2 = genfis2(trainingData(:,1:end-1),trainingData(:,end),radii2);
for i = 1:size(fis2.output.mf, 2)        
    constt = fis2.output.mf(i).params(end);
    fis2.output.mf(i).type = 'constant';
    fis2.output.mf(i).params = constt;       
end
%% Training

opt = anfisOptions;
opt.InitialFIS = fis1;
opt.EpochNumber = 100;
opt.ValidationData = validationData;
opt.OptimizationMethod = 1;
%% 1st
[trnFis,trnError,~,valFis,valError] = anfis(trainingData, opt);

Y=evalfis(chkData(:,1:end-1),valFis);

Y=round(Y);

%% 1) Fuzzy MFs

figure;
hold on;
plotmf(valFis, 'input', 1);

figure;
hold on;
plotmf(valFis, 'input', 2);

figure;
hold on;
plotmf(valFis, 'input', 3);

%% 2) Learning Curve

figure;
hold on;
plot(trnError);
plot(valError);
ylabel('Error');
xlabel('Epochs');
legend('Training Curve', 'Validation Curve');

%% 3) Prediction Diagrams

figure;
hold on;
plot(1:floor(m*0.2), chkData(:,end));
plot(1:floor(m*0.2), Y');
ylabel('Survival');
xlabel('Sample');
legend('Data', 'Prediction');

predictionError = chkData(:, end) - Y;
figure;
hold on;
plot(1:floor(m*0.2), predictionError');
ylabel('Prediction Errors');
xlabel('Sample');

%% 4) Performance Indicators

cfmat1 = confusionmat(chkData(:, size(data, 2)), Y);
    
OA1 = trace(cfmat1)/size(chkData, 1);

PA1 = zeros(2,1);
UA1 = zeros(2,1);
for i = 1:2
   PA1(i) = cfmat1(i,i)/sum(cfmat1(i,:)); 
end
for i = 1:2
   UA1(i) = cfmat1(i,i)/sum(cfmat1(:,i)); 
end

sumMulSum = 0;
for i = 1:2
   sumMulSum = sumMulSum + sum(cfmat1(i,:))*sum(cfmat1(:,i));   
end
K1 = ((size(chkData, 1)^2)*OA1 - sumMulSum)/((size(chkData, 1))^2 - sumMulSum);

disp('--------------')
disp(['Overall Accuracy: ' num2str(OA1)])
disp(['K: ' num2str(K1)])
disp('--------------')


%% 2nd
opt.InitialFIS = fis2;
[trnFis,trnError,~,valFis,valError] = anfis(trainingData, opt);

Y=evalfis(chkData(:,1:end-1),valFis);

Y=round(Y);

%% 1) Fuzzy MFs

figure;
hold on;
plotmf(valFis, 'input', 1);

figure;
hold on;
plotmf(valFis, 'input', 2);

figure;
hold on;
plotmf(valFis, 'input', 3);

%% 2) Learning Curve

figure;
hold on;
plot(trnError);
plot(valError);
ylabel('Error');
xlabel('Epochs');
legend('Training Curve', 'Validation Curve');

%% 3) Prediction Diagrams

figure;
hold on;
plot(1:floor(m*0.2), chkData(:,end));
plot(1:floor(m*0.2), Y');
ylabel('Survival');
xlabel('Sample');
legend('Data', 'Prediction');

predictionError = chkData(:, end) - Y;
figure;
hold on;
plot(1:floor(m*0.2), predictionError');
ylabel('Prediction Errors');
xlabel('Sample');

%% 4) Performance Indicators

cfmat2 = confusionmat(chkData(:, size(data, 2)), Y);
    
OA2 = trace(cfmat2)/size(chkData, 1);

PA2 = zeros(2,1);
UA2 = zeros(2,1);
for i = 1:2
   PA2(i) = cfmat2(i,i)/sum(cfmat2(i,:)); 
end
for i = 1:2
   UA2(i) = cfmat2(i,i)/sum(cfmat2(:,i)); 
end

sumMulSum = 0;
for i = 1:2
   sumMulSum = sumMulSum + sum(cfmat2(i,:))*sum(cfmat2(:,i));   
end
K2 = ((size(chkData, 1)^2)*OA2 - sumMulSum)/((size(chkData, 1))^2 - sumMulSum);

disp('--------------')
disp(['Overall Accuracy: ' num2str(OA2)])
disp(['K: ' num2str(K2)])
disp('--------------')