function [dh,dv] = obstacleDistance(x,y)

    if y <= 5
        dh = 10 - x;
    elseif y <= 6
        dh = 11 - x;
    elseif y <= 7
        dh = 12 - x;
    else
        dh = 15;
    end
        
    if x <= 10
        dv = y;
    elseif x <= 11
        dv = y - 5;
    elseif x <= 12
        dv = y - 6;
    else
        dv = y - 7;
    end  

end

