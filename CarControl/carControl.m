carcontrol = readfis('carcontrolBetter.fis');

u = 0.05;
x1 = 9.1;
y1 = 4.3;
theta11 = 0;
theta12 = 45;
theta13 = 90;
xfinal = 15;
yfinal = 7.2;

n = 162;
x = zeros(n,1);
y = zeros(n,1);

x(1) = x1;
y(1) = y1;

theta = theta13;

i = 1;

while x < 15 & i < n
    [dh,dv] = obstacleDistance(x(i),y(i));
    dh = min(1,max(0,dh/15));
    dv = min(1,max(0,dv/7.2));
    detheta = evalfis([dv dh theta], carcontrol);
    theta = min(180, max(-180, theta + detheta));
    
    x(i+1)=x(i)+cosd(theta)*u;  
    y(i+1)=y(i)+sind(theta)*u;
    
    i = i + 1;
end

figure;
hold on;
axis([0 15 0 9]);
plot(x, y);
plot([10 10 11 11 12 12 15], [0 5 5 6 6 7 7]);

error = y(n)-yfinal
