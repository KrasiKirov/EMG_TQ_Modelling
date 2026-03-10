function [ramp_signal]=gen_ramp_and_hold(initial_value,final_value,slope,start_time,finish_time)



% if initial_value==final_value
%     error('Initial and final values are identical')
% end


fs=1000;

if initial_value<final_value
mid=(initial_value:slope/fs:final_value-slope/fs);
size(mid)
else
    mid=(initial_value:slope/fs:final_value+slope/fs);
    size(mid)
end

head=ones(1,start_time*fs)*initial_value;
tail=ones(1,finish_time*fs)*final_value;
amp=[head,mid,tail];
total_duration=start_time+finish_time+abs(initial_value-final_value)/abs(slope);
time=(0:total_duration/size(amp,2):total_duration-total_duration/size(amp,2));


ramp_signal=[time;amp];
save('ramp_signal');

