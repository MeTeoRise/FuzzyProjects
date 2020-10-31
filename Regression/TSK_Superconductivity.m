close all;
clear all;

%% Data preparation

data = readtable("superconductivity.csv");
data = table2array(data);
[m,n] = size(data);

% trainingData = data(1:floor(0.6*m),:);
% validationData = data(ceil(0.6*m):floor(0.8*m),:);
% chkData = data(ceil(0.8*m):m,:);

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

Fno = [5 8 12 15];
radiis = [0.2 0.4 0.6 0.8];

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

%% Select The Best Parameters

minMSE = min(min(cvMSE));
[FeatureNumber,radiiIndx]=find(cvMSE==minMSE);

% FeatureNumber = 3;
% radiiIndx = 1;

features(1:Fno(FeatureNumber)) = ranks(1:Fno(FeatureNumber))';
trainingDataOptimal = trainingData(:, features(1:Fno(FeatureNumber)));
radii = radiis(radiiIndx);
fismat  = genfis2(trainingDataOptimal, trainingData(:, size(data, 2)), radii);

opt = anfisOptions('InitialFIS',fismat , 'ValidationData', validationData(:, [features size(data, 2)]), 'OptimizationMethod', 1, 'EpochNumber', 50);
[trnFis,trnError,~,valFis,valError] = anfis(trainingData(:, [features size(data, 2)]), opt);

Y = evalfis(chkData(:,features(1:Fno(FeatureNumber))), valFis);

R2 = 1-sum((Y-chkData(:,end)).^2)/sum((chkData(:,end)-mean(chkData(:,end))).^2);

%% 1) Fuzzy MFs

figure;
hold on;
plotmf(valFis, 'input', 1);

figure;
hold on;
plotmf(valFis, 'input', 2);

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
plot(1:ceil(m*0.2), chkData(:,end));
plot(1:ceil(m*0.2), Y');
ylabel('Sound pressure level (Decibels)');
xlabel('Sample');
legend('Data', 'Prediction');

predictionError = chkData(:, end) - Y;
figure;
hold on;
plot(1:ceil(m*0.2), predictionError');
ylabel('Prediction Errors');
xlabel('Sample');

%% 4) Performance Indicators

sigmax= sum(((chkData(:,end)-mean(chkData(:,end))).^2))/ceil(m*0.2);

RMSE=sqrt(mse(Y,chkData(:,end)));

NMSE = RMSE.^2/sigmax;

NDEI = sqrt(NMSE);

disp('--------------')
disp(['RMSE: ' num2str(RMSE)])
disp(['NMSE: ' num2str(NMSE)])
disp(['NDEI: ' num2str(NDEI)])
disp(['R2: ' num2str(R2)])
disp('--------------')

%% predfun

function yfit = predfun(Xtrain,ytrain,Xtest)

global ii
global radiis

radii = radiis(ii);

fismat = genfis2(Xtrain, ytrain, radii);

opt = anfisOptions('InitialFIS', fismat, 'OptimizationMethod', 1, 'EpochNumber', 50);
fis = anfis([Xtrain ytrain], opt);

yfit = evalfis(Xtest, fis);

end