function [piecewise_const_signal]=gen_piecewise_const(max_pf,max_df,min_pulse_hold,max_pulse_hold,duration)


max_pf=-1*max_pf;
fs=1000;
n=ceil(duration/min_pulse_hold);

s=0.02;
for i =1:n;

   
            rand_amp=-max_pf+(max_df+max_pf).*rand;
            rand_time=min_pulse_hold+(max_pulse_hold-min_pulse_hold).*rand;
            rand_samples=floor((floor(rand_time*1000)/1000).*fs);
            a=rand_amp.*ones(1,rand_samples);
   
    
            
            s=[s a];
end

length=duration*fs;
for j=1:length;
    
    sig(j)=s(j);
end
    
% filtOrder = 2; fCutOffHz = 3; FsHz = 1000;
% myFilter = design(fdesign.lowpass('N,Fc',filtOrder,fCutOffHz,FsHz));

myFilter=1/500*ones(500,1);

piecewise_const_signal=filter(myFilter,1,sig);
time=(0:1/fs:size(piecewise_const_signal,2)/fs-1/fs);
piecewise_const_signal=[time;piecewise_const_signal];










