close all;
clear all;

%% Data preparation
data = load("airfoil_self_noise.dat");

[m,n] = size(data);

% for j = 1:n
%     maxv = max(data(:, j));
%     minv = min(data(:, j));
%     for i = 1:m     
%         data(i, j) = (data(i, j) - minv)/(maxv - minv);
%     end
% end

% data = normalize(data, 'range', [-1 1]);

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

% trainingData = data(1:floor(0.6*m),:);
% validationData = data(ceil(0.6*m):floor(0.8*m),:);
% chkData = data(ceil(0.8*m):m,:);
%% Generate fis

fis = genfis1(data, 3, 'gbellmf', 'linear');
%% Training

opt = anfisOptions;
opt.InitialFIS = fis;
opt.EpochNumber = 100;
opt.ValidationData = validationData;
options.ErrorGoal = 0.1;
opt.OptimizationMethod = 1;

[trnFis,trnError,~,valFis,valError] = anfis(trainingData, opt);

Y=evalfis(chkData(:,1:end-1),valFis);

R2 = 1-(sum((Y-chkData(:,end)).^2)/sum((chkData(:,end)-mean(chkData(:,end))).^2));

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

figure;
hold on;
plotmf(valFis, 'input', 4);

figure;
hold on;
plotmf(valFis, 'input', 5);

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

sigmax = sum(((chkData(:,end)-mean(chkData(:,end))).^2))/ceil(m*0.2);

RMSE = sqrt(mse(Y,chkData(:,end)));

NMSE = RMSE.^2/sigmax;

NDEI = sqrt(NMSE);

disp('--------------')
disp(['RMSE: ' num2str(RMSE)])
disp(['NMSE: ' num2str(NMSE)])
disp(['NDEI: ' num2str(NDEI)])
disp(['R2: ' num2str(R2)])
disp('--------------')
