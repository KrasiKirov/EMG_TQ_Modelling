% close all

%% Code to read MVC and passive torque data
trialNo = 6; %0;
fileName = 'ET_250425.flb'; %'HM_110425.flb'; %'CH_151224.flb'; 'AG_070225.flb'; 

set(0,'DefaultFigureWindowStyle','docked')

POS = 1;
TQ = 2;
Ts = 0.001;

trial_duration = 30;  %90;

t1 = 0; n1 = fix(t1/Ts + 1);
t2 = trial_duration-Ts; n2 = fix(t2/Ts);

filePath = 'G:\Users\Ehsan\Data\'; 

data = flb2mat([filePath,fileName],'read_all'); clc;

mean_tq = mean(data{1,trialNo}(n1:n2,TQ)).dataSet;
disp(['Mean Passive TQ = ', num2str(mean_tq), 'Nm'])

init_tq = data{1,trialNo}(1,TQ).dataSet;
disp(['Initial TQ = ', num2str(init_tq), 'Nm'])

figure; plot(data{1,trialNo}(n1:n2,TQ)); ylabel('Torque (Nm)'); grid on
title(['Mean Passive TQ = ', num2str(mean_tq), 'Nm. ', 'Initial TQ = ', num2str(init_tq), 'Nm.'])

disp(data{1,trialNo}.comment)

mvc_tq = min(data{1,trialNo}(:,TQ)).dataSet;
disp(['MVC TQ = ', num2str(mvc_tq), 'Nm'])

figure; 
subplot(5,1,1); plot(data{1,trialNo}(n1:n2,TQ)); ylabel('Torque (Nm)'); grid on
subplot(5,1,2); plot(data{1,trialNo}(n1:n2,3)); ylabel('GL (mV)'); grid on;
subplot(5,1,3); plot(data{1,trialNo}(n1:n2,4)); ylabel('TA (mV)'); grid on;
subplot(5,1,4); plot(data{1,trialNo}(n1:n2,5)); ylabel('GM (mV)'); grid on;
subplot(5,1,5); plot(data{1,trialNo}(n1:n2,6)); ylabel('Sol (mV)'); grid on;
xAxisPanZoom