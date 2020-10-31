tin = 0:0.01:30;
sig = [150*ones(1, 1000) 100*ones(1, 1000) 150*ones(1, 1001)];
s1 = timeseries(sig, tin);

figure
plot(s1);

tin = 0:0.01:30;
sig = [0.15*(1:1000) 150*ones(1,1000) 0.15*(1000:-1:0)];
s2 = timeseries(sig, tin);

figure
plot(s2);

tin = 0:0.01:30;
sig = [150*ones(1, 3001)];
s3 = timeseries(sig, tin);

figure
plot(s3);

tin = 0:0.01:30;
sig = [0*ones(1, 1000) 1*ones(1,1000) 0*ones(1,1001)];
s3dist = timeseries(sig, tin);

figure
plot(s3dist);
