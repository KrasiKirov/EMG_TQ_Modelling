function [ramp_signal min_period]=gen_periodic_ramp_and_hold(initial_value,final_value,slope,cycle,duration)

if initial_value>final_value 
    if slope>0
        slope=slope*-1;
    end
end

if initial_value==final_value
    error('Initial and final values are identical')
end


fs=1000;

if initial_value<final_value
mid=(initial_value:slope/fs:final_value-slope/fs);
else
    mid=(initial_value:slope/fs:final_value+slope/fs);
end

ncycle=ceil(duration*(1/cycle));
start_time=(1/(2*(1/cycle))-abs(initial_value-final_value)/abs(slope))/2;
finish_time=start_time;
head=ones(1,ceil(start_time*fs))*initial_value;
tail=ones(1,ceil(finish_time*fs))*final_value;
amp=[head,mid,tail];
amp_single_pulse=[amp flip(amp)]; 
amp=repmat(amp_single_pulse,1,ncycle);


for n=1:duration*fs
    trimmed_amp(n)=amp(n);
end


total_duration=(start_time+finish_time+abs(initial_value-final_value)/abs(slope))*2;

min_period=(2*abs(initial_value-final_value))/abs(slope);

time=(0:1/fs:size(trimmed_amp,2)/fs-1/fs);
ramp_signal=[time;trimmed_amp];


