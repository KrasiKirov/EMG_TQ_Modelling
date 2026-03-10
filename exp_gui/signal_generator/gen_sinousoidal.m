function [sin_signal]=gen_sinousoidal(amplitude,frequency,duration)
fs=1000;
time=(0:1/fs:duration-1/fs);
amp=amplitude*sin(time*2*pi*frequency);
sin_signal=[time;amp];
save('sin_signal');
