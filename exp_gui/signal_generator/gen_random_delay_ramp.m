function [Sig]=gen_random_delay_ramp(initial_value,final_value,slope,min_delay,max_delay,duration)


fs=1000;

a=gen_ramp_and_hold(initial_value,final_value,slope,0,0);
a=a(2,:);
s=size(a,2);
s=s/fs;
s=ceil(s+(min_delay).*2);
n=ceil(duration/s);


sig=[initial_value.*(ones(1,3000)) a];


for i=1:n
    
rand_delay=min_delay+(max_delay-min_delay).*rand;
rand_delay_s=ceil(rand_delay.*fs);
hold=ones(1,rand_delay_s).*final_value;
sig=[sig hold];
a_flip=flip(a);
sig=[sig a_flip];
rand_delay=min_delay+(max_delay-min_delay).*rand;
rand_delay_s=ceil(rand_delay.*fs);
hold=ones(1,rand_delay_s).*initial_value;
sig=[sig hold];
sig=[sig a];
end

length=duration*fs+1;

for j=1:length;

    if j<length
    Sig(j)=sig(j);
    end
end


time=(0:1/fs:duration-1/fs);

Sig=[time;Sig];


