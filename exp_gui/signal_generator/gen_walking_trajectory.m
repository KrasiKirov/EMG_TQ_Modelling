function [walk_signal]=gen_walking_trajectory(ROM_low,ROM_high,cycle_duration,duration)

fs=1000;
amp_0=load('walk_amp_interpolated.mat','walk_amp_interpolated');
amp_0=pi/180*amp_0.walk_amp_interpolated;
no_cycle=ceil(duration/cycle_duration);
no_sample_per_cycle=cycle_duration.*fs;

%amp_0=resample(amp_0,no_sample_per_cycle,size(amp_0,2));

amp_1=repmat(amp_0,1,no_cycle);
amp_1=resample(amp_1,no_sample_per_cycle.*no_cycle,size(amp_1,2));

trimmed_amp=0;

for n=(1:1:(fs*(duration)))
    trimmed_amp(n)=amp_1(n);
end

%duration_new=size(amp_1,2)/fs;

time=(0:1/fs:duration-1/fs);

% scaling

ROM_low;
ROM_high;
f=zeros(1,size(trimmed_amp,2));
for n=1:size(trimmed_amp,2)
    if trimmed_amp(n)>0
        f(n)=trimmed_amp(n)*ROM_high/max(trimmed_amp);
    else
        f(n)=trimmed_amp(n)*ROM_low/min(trimmed_amp); 
    end
end

amp=f;
duration
size(amp)
walk_signal=[time;amp];





