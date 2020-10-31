close all;
clear all;

%% Data preparation

data = readtable("data.csv");
data = table2cell(data);
data = data(:,2:end);
data = cell2mat(data);

[m,n] = size(data);

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

[ranks, weights] = relieff(trainingData(:,1:(size(data, 2) - 1)), trainingData(:, size(data, 2)), 10);

c = cvpartition(trainingData(:,size(data, 2)), 'Kfold', 5);

global radiis

Fno = [4 7 10 13];
radiis = [0.2 0.35 0.5 0.65];

global ii 
ii = 0;

%% Cross Validation

for i = 1:length(Fno)
    
    features(1:Fno(i)) = ranks(1:Fno(i));
        
    for j = 1:length(radiis)
      
        ii = ii+1;
        
        cvMSE(i,j) = crossval('mse', trainingData(:, features), trainingData(:, size(data, 2)), 'predfun', @predfun, 'partition', c);
       
    end
    ii = 0;
end

minMSE = min(min(cvMSE));
[FeatureNumber,radiiIndx]=find(cvMSE==minMSE);

% FeatureNumber = 4;
% radiiIndx = 1;

features(1:Fno(FeatureNumber)) = ranks(1:Fno(FeatureNumber))';
trainingDataOptimal = trainingData(:, features(1:Fno(FeatureNumber)));
radii = radiis(radiiIndx);
fismat  = genfis2(trainingDataOptimal, trainingData(:, size(data, 2)), radii);
for i = 1:size(fismat.output.mf, 2)        
    constt = fismat.output.mf(i).params(end);
    fismat.output.mf(i).type = 'constant';
    fismat.output.mf(i).params = constt;        
end

opt = anfisOptions('InitialFIS',fismat , 'ValidationData', validationData(:, [features size(data, 2)]), 'OptimizationMethod', 1, 'EpochNumber', 50);
[trnFis,trnError,~,valFis,valError] = anfis(trainingData(:, [features size(data, 2)]), opt);

Y = evalfis(chkData(:,features(1:Fno(FeatureNumber))), valFis);
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
ylabel('Status');
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


%% predfun

function yfit = predfun(Xtrain,ytrain,Xtest)

global ii
global radiis

radii = radiis(ii);

fismat = genfis2(Xtrain, ytrain, radii);
for i = 1:size(fismat.output.mf, 2)        
    constt = fismat.output.mf(i).params(end);
    fismat.output.mf(i).type = 'constant';
    fismat.output.mf(i).params = constt;        
end

opt = anfisOptions('InitialFIS', fismat, 'OptimizationMethod', 1, 'EpochNumber', 35);
fis = anfis([Xtrain ytrain], opt);

yfit = evalfis(Xtest, fis);

end