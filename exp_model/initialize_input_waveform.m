n_angles = 8;
maxPF = -0.5;  %-- rad
maxDF = +0.2;  %-- rad

sampleTime = 0.001;
endTime = 5;
numberOfSamples = endTime * 1/sampleTime + 1;
timeVector = (0:numberOfSamples) * sampleTime;
c= timeseries(timeVector*0,timeVector);
waveform = c;