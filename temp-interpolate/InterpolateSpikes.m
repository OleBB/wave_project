function eta_out=InterpolateSpikes(eta_in,Threshold)
%Removes and interpolates spikes.
%(code fra Anne Raustoel 2012, with some changes)
% eta_in is a Nx1 vector with surface elevation data,
% t is a Nx1 vector with the time.
% Threshold is the maximum slope between two datapoints.
% Interpolation method is 'pchip' default

E = length(eta_in);
t = 1: +1 :E;
t = transpose(t);


eta_out=eta_in;

np=0; % Total number of points removed.
counter=0; % Total number of passes


while counter<20
    counter=counter+1;
    npr=0; % Number of points removed in each pass.
    
    for i=3:length(eta_in)-3
        if eta_out(i)<0 % only looks at values below the mean.
            if abs((eta_out(i+1)-eta_out(i))/(t(i+1)-t(i)))>Threshold
                eta_out(i)=NaN;
                eta_out(i+1)=NaN;
                eta_out(i+2)=NaN;
                eta_out(i-1)=NaN;
                eta_out(i-2)=NaN;
                npr=npr+1;
            end
        end
    end

    np=np+npr;
    index = find(isnan(eta_out));
    eta_out(index)=interp1(find(~isnan(eta_out)),eta_out(~isnan(eta_out)), index, 'pchip');
end
end
