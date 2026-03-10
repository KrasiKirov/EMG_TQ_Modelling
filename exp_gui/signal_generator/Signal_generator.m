function varargout = Signal_generator(varargin)
% SIGNAL_GENERATOR MATLAB code for Signal_generator.fig
%      SIGNAL_GENERATOR, by itself, creates a new SIGNAL_GENERATOR or raises the existing
%      singleton*.
%
%      H = SIGNAL_GENERATOR returns the handle to a new SIGNAL_GENERATOR or the handle to
%      the existing singleton*.
%
%      SIGNAL_GENERATOR('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in SIGNAL_GENERATOR.M with the given input arguments.
%
%      SIGNAL_GENERATOR('Property','Value',...) creates a new SIGNAL_GENERATOR or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before Signal_generator_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to Signal_generator_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help Signal_generator

% Last Modified by GUIDE v2.5 19-Sep-2019 14:05:16

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @Signal_generator_OpeningFcn, ...
                   'gui_OutputFcn',  @Signal_generator_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before Signal_generator is made visible.
function Signal_generator_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to Signal_generator (see VARARGIN)

% Choose default command line output for Signal_generator
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes Signal_generator wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = Signal_generator_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in sin_TAB.
function sin_TAB_Callback(hObject, eventdata, handles)

%the sine_TAB button will make sin_PANEL,the only visible panel

set(handles.sin_PANEL, 'visible','on');
set(handles.ramp_PANEL, 'visible','off');
set(handles.walk_PANEL, 'visible','off');
set(handles.p_ramp_PANEL, 'visible','off');
set(handles.r_ramp_PANEL, 'visible','off');
set(handles.piecewise_PANEL, 'visible','off');

% --- Executes on button press in ramp_TAB.
function ramp_TAB_Callback(hObject, eventdata, handles)

%the ramp_TAB button will make ramp_PANEL,the only visible panel

set(handles.sin_PANEL, 'visible','off');
set(handles.ramp_PANEL, 'visible','on');
set(handles.walk_PANEL, 'visible','off');
set(handles.p_ramp_PANEL, 'visible','off');
set(handles.piecewise_PANEL, 'visible','off');
set(handles.r_ramp_PANEL, 'visible','off');

% --- Executes on button press in radiobutton1.
function radiobutton1_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton1


% --- Executes on selection change in popupmenu1.
function popupmenu1_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu1 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu1


% --- Executes during object creation, after setting all properties.
function popupmenu1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in walk_TAB.
function walk_TAB_Callback(hObject, eventdata, handles)

%the walk_TAB button will make walk_PANEL,the only visible panel

set(handles.sin_PANEL, 'visible','off');
set(handles.ramp_PANEL, 'visible','off');
set(handles.walk_PANEL, 'visible','on');
set(handles.p_ramp_PANEL, 'visible','off');
set(handles.r_ramp_PANEL, 'visible','off');
set(handles.piecewise_PANEL, 'visible','off');


function sin_frequency_Callback(hObject, eventdata, handles)
% hObject    handle to sin_frequency (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of sin_frequency as text
%        str2double(get(hObject,'String')) returns contents of sin_frequency as a double


% --- Executes during object creation, after setting all properties.
function sin_frequency_CreateFcn(hObject, eventdata, handles)
% hObject    handle to sin_frequency (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function sin_amplitude_Callback(hObject, eventdata, handles)
% hObject    handle to sin_amplitude (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of sin_amplitude as text
%        str2double(get(hObject,'String')) returns contents of sin_amplitude as a double


% --- Executes during object creation, after setting all properties.
function sin_amplitude_CreateFcn(hObject, eventdata, handles)
% hObject    handle to sin_amplitude (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in sin_GENERATE.
function sin_GENERATE_Callback(hObject, eventdata, handles)

%acquiring the parameter values from the gui 

sin_amplitude_val=str2double(get(handles.sin_amplitude,'string'));
sin_frequency_val=str2double(get(handles.sin_frequency,'string'));
sin_duration_val=str2double(get(handles.sin_duration,'string'));

%generating signal value using gen_sinusoidal
sin_ans=gen_sinousoidal(sin_amplitude_val,sin_frequency_val,sin_duration_val);
sin_amp=sin_ans(2,:);
sin_time=sin_ans(1,:);

%plotting
plot(sin_time,sin_amp,'Linewidth',2,'Parent',handles.sin_AXES);
grid(handles.sin_AXES, 'on');
xlabel('Time (s)','Parent',handles.sin_AXES);
ylabel('Position (rad)','Parent',handles.sin_AXES);

if sin_duration_val > 10
    
set(handles.sin_AXES,'XLim',[0,10]);

end

set(gca,'FontSize',20);

%saving the signal along with its parameters as a struct

saved_sin=struct('Sig',sin_ans,'sin_freq',sin_frequency_val,'sin_amp',sin_amplitude_val,'Duration',sin_duration_val);

%setting saved_sin struct in the app data in order for it ti accessble by
%other functions (INSTEAD OF USING GLOBAL !)

setappdata(0,'saved_sin',saved_sin);

%constructing the base file name
p_1=num2str(sin_amplitude_val);
p_2=num2str(sin_frequency_val);
p_3=num2str(sin_duration_val);
prefix=['sin__' p_1 '__' p_2 '__' p_3 '__' ];

%replacing '.' (Decimal separator) and '-' (Negative) characters with '_','N' 

prefix=strrep(prefix,'.','_');
prefix=strrep(prefix,'-','N');
setappdata(0,'sin_prefix',prefix);


% --- Executes on button press in sin_SAVE.
function sin_SAVE_Callback(hObject, eventdata, handles)

saved_sin=getappdata(0,'saved_sin');


name=get(handles.sin_file_name,'string');
sin_prefix=char(getappdata(0,'sin_prefix'));
name=char(name);
name=[sin_prefix name];
% save(strcat('G:\Users\Ali_R\GUI_draft\saved\sin\',name),'saved_sin');

set(handles.saved_PANEL,'visible','on');
set(handles.text51,'string',strcat('G:\Users\Ali_R\GUI_draft\saved\sin\',name));




function sin_duration_Callback(hObject, eventdata, handles)
% hObject    handle to sin_duration (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of sin_duration as text
%        str2double(get(hObject,'String')) returns contents of sin_duration as a double


% --- Executes during object creation, after setting all properties.
function sin_duration_CreateFcn(hObject, eventdata, handles)
% hObject    handle to sin_duration (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function ramp_initial_value_Callback(hObject, eventdata, handles)
% hObject    handle to ramp_initial_value (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of ramp_initial_value as text
%        str2double(get(hObject,'String')) returns contents of ramp_initial_value as a double


% --- Executes during object creation, after setting all properties.
function ramp_initial_value_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ramp_initial_value (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function ramp_final_value_Callback(hObject, eventdata, handles)
% hObject    handle to ramp_final_value (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of ramp_final_value as text
%        str2double(get(hObject,'String')) returns contents of ramp_final_value as a double


% --- Executes during object creation, after setting all properties.
function ramp_final_value_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ramp_final_value (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function ramp_slope_Callback(hObject, eventdata, handles)
% hObject    handle to ramp_slope (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of ramp_slope as text
%        str2double(get(hObject,'String')) returns contents of ramp_slope as a double


% --- Executes during object creation, after setting all properties.
function ramp_slope_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ramp_slope (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function ramp_initial_hold_duration_Callback(hObject, eventdata, handles)
% hObject    handle to ramp_initial_hold_duration (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of ramp_initial_hold_duration as text
%        str2double(get(hObject,'String')) returns contents of ramp_initial_hold_duration as a double


% --- Executes during object creation, after setting all properties.
function ramp_initial_hold_duration_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ramp_initial_hold_duration (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function ramp_final_hold_duration_Callback(hObject, eventdata, handles)
% hObject    handle to ramp_final_hold_duration (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of ramp_final_hold_duration as text
%        str2double(get(hObject,'String')) returns contents of ramp_final_hold_duration as a double


% --- Executes during object creation, after setting all properties.
function ramp_final_hold_duration_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ramp_final_hold_duration (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in ramp_GENERATE.
function ramp_GENERATE_Callback(hObject, eventdata, handles)
fs=1000;
initial_value_val=str2double(get(handles.ramp_initial_value,'string'));
final_value_val=str2double(get(handles.ramp_final_value,'string'));

slope_val=str2double(get(handles.ramp_slope,'string'));
ramp_and_hold_initial_hold_duration_val=str2double(get(handles.ramp_initial_hold_duration,'string'));
ramp_and_hold_final_hold_duration_val=str2double(get(handles.ramp_final_hold_duration,'string'));

if initial_value_val>final_value_val
    if slope_val>0
      slope_val=slope_val.*-1;
    end
end

if initial_value_val~=final_value_val
    
        ramp_ans=gen_ramp_and_hold(initial_value_val,final_value_val,slope_val,ramp_and_hold_initial_hold_duration_val,ramp_and_hold_final_hold_duration_val);
else
        ramp_ans=zeros(1,10000);
        time_f=(0:1/1000:10-1/1000);
        ramp_ans=[time_f;ramp_ans];
end

    
ramp_amp=ramp_ans(2,:);
ramp_time=ramp_ans(1,:);
duration=size(ramp_amp,2)/fs;







plot(ramp_time,ramp_amp,'Linewidth',2,'Parent',handles.ramp_AXES);
grid(handles.ramp_AXES, 'on');
max_l=max(abs(final_value_val),abs(initial_value_val))*1.5;
xlabel('Time (s)','Parent',handles.ramp_AXES);
ylabel('Position (rad)','Parent',handles.ramp_AXES);

ramp_saved=struct('Sig',ramp_ans,'Initial_value',initial_value_val,'Final_value',final_value_val,'Slope',slope_val,'Initial_hold',ramp_and_hold_initial_hold_duration_val,'Final_hold',ramp_and_hold_final_hold_duration_val,'Duration',duration);
setappdata(0,'ramp_saved',ramp_saved);

p_1=num2str(initial_value_val);
p_2=num2str(final_value_val);
p_3=num2str(slope_val);
p_4=num2str(ramp_and_hold_initial_hold_duration_val);
p_5=num2str(ramp_and_hold_final_hold_duration_val);
p_6=num2str(duration);

prefix=['ramp&hold__' p_1 '__' p_2 '__' p_3 '__' p_4 '__' p_5 '__' p_6 '__'];

prefix=strrep(prefix,'.','_');
prefix=strrep(prefix,'-','N');
setappdata(0,'ramp_prefix',prefix);

% --- Executes on button press in ramp_SAVE.
function ramp_SAVE_Callback(hObject, eventdata, handles)

ramp_saved=getappdata(0,'ramp_saved');


name=get(handles.ramp_file_name,'string');
ramp_prefix=char(getappdata(0,'ramp_prefix'));
name=char(name);
name=[ramp_prefix name];
save(strcat('G:\Users\Ali_R\GUI_draft\saved\ramp\',name),'ramp_saved');


set(handles.saved_PANEL,'visible','on');
set(handles.text51,'string',strcat('G:\Users\Ali_R\GUI_draft\saved\ramp_saved\',name));





function walk_min_angle_Callback(hObject, eventdata, handles)
% hObject    handle to walk_min_angle (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of walk_min_angle as text
%        str2double(get(hObject,'String')) returns contents of walk_min_angle as a double


% --- Executes during object creation, after setting all properties.
function walk_min_angle_CreateFcn(hObject, eventdata, handles)
% hObject    handle to walk_min_angle (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function walking_cycle_duration_Callback(hObject, eventdata, handles)
% hObject    handle to walking_cycle_duration (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of walking_cycle_duration as text
%        str2double(get(hObject,'String')) returns contents of walking_cycle_duration as a double


% --- Executes during object creation, after setting all properties.
function walking_cycle_duration_CreateFcn(hObject, eventdata, handles)
% hObject    handle to walking_cycle_duration (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function walk_duration_Callback(hObject, eventdata, handles)
% hObject    handle to walk_duration (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of walk_duration as text
%        str2double(get(hObject,'String')) returns contents of walk_duration as a double


% --- Executes during object creation, after setting all properties.
function walk_duration_CreateFcn(hObject, eventdata, handles)
% hObject    handle to walk_duration (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function walk_max_angle_Callback(hObject, eventdata, handles)
% hObject    handle to walk_max_angle (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of walk_max_angle as text
%        str2double(get(hObject,'String')) returns contents of walk_max_angle as a double


% --- Executes during object creation, after setting all properties.
function walk_max_angle_CreateFcn(hObject, eventdata, handles)
% hObject    handle to walk_max_angle (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in walk_GENERATE.
function walk_GENERATE_Callback(hObject, eventdata, handles)
ROM_low=str2double(get(handles.walk_min_angle,'string'));
ROM_high=str2double(get(handles.walk_max_angle,'string'));
walking_trajectory_period_val=str2double(get(handles.walking_cycle_duration,'string'));
walking_trajectory_duration_val=str2double(get(handles.walk_duration,'string'));
walk_ans=gen_walking_trajectory(ROM_low,ROM_high,walking_trajectory_period_val,walking_trajectory_duration_val);
walk_amp=walk_ans(2,:);
walk_time=walk_ans(1,:);
plot(walk_time,walk_amp,'Linewidth',2,'Parent',handles.walk_AXES);
grid(handles.walk_AXES, 'on');
if walking_trajectory_duration_val > 10
    
set(handles.walk_AXES,'XLim',[0,10]);

end

xlabel('Time (s)','Parent',handles.walk_AXES);
ylabel('Position (rad)','Parent',handles.walk_AXES);


walk_saved=struct('Sig',walk_ans,'walk_ROM_low',ROM_low,'walk_ROM_high',ROM_high,'walk_period',walking_trajectory_period_val,'Duration',walking_trajectory_duration_val);
setappdata(0,'walk_saved',walk_saved);

p_1=num2str(ROM_low);
p_2=num2str(ROM_high);
p_3=num2str(walking_trajectory_period_val);
p_4=num2str(walking_trajectory_duration_val);
prefix=['walk__' p_1 '__' p_2 '__' p_3 '__' p_4 '__'];

prefix=strrep(prefix,'.','_');
prefix=strrep(prefix,'-','N');
setappdata(0,'walk_prefix',prefix);

% --- Executes on button press in walk_SAVE.
function walk_SAVE_Callback(hObject, eventdata, handles)



walk_saved=getappdata(0,'walk_saved');


name=get(handles.walk_file_name,'string');
walk_prefix=char(getappdata(0,'walk_prefix'));
name=char(name);
name=[walk_prefix name];

save(strcat('G:\Users\Ali_R\GUI_draft\saved\walk\',name),'walk_saved');

set(handles.saved_PANEL,'visible','on');
set(handles.text51,'string',strcat('G:\Users\Ali_R\GUI_draft\saved\walk\',name));

% --- Executes on button press in p_ramp_TAB.
function p_ramp_TAB_Callback(hObject, eventdata, handles)
set(handles.piecewise_PANEL, 'visible','off');
set(handles.sin_PANEL, 'visible','off');
set(handles.ramp_PANEL, 'visible','off');
set(handles.walk_PANEL, 'visible','off');
set(handles.p_ramp_PANEL, 'visible','on');
set(handles.r_ramp_PANEL, 'visible','off');
% --- Executes on button press in radiobutton1
% --- Executes on button press in p_ramp_GENERATE.

function p_ramp_GENERATE_Callback(hObject, eventdata, handles)

initial_value_val=str2double(get(handles.p_ramp_initial_value,'string'));
final_value_val=str2double(get(handles.p_ramp_final_value,'string'));
slope_val=str2double(get(handles.p_ramp_slope,'string'));
period=str2double(get(handles.p_ramp_cycle_period,'string'));
duration=str2double(get(handles.p_ramp_duration,'string'));

[periodic_ramp_ans min_period]=gen_periodic_ramp_and_hold(initial_value_val,final_value_val,slope_val,period,duration);

periodic_ramp_amp=periodic_ramp_ans(2,:);
periodic_ramp_time=periodic_ramp_ans(1,:);

plot(periodic_ramp_time,periodic_ramp_amp,'Linewidth',2,'Parent',handles.p_ramp_AXES);
grid(handles.p_ramp_AXES, 'on');
xlabel('Time (s)','Parent',handles.p_ramp_AXES);
ylabel('Position (rad)','Parent',handles.p_ramp_AXES);


if duration > 10
    
set(handles.p_ramp_AXES,'XLim',[0,10]);


end


min_period_str=num2str(min_period);
min_period_str=[min_period_str ' (s)'];

set(handles.text41,'String',min_period_str);

if period < min_period
set(handles.p_ramp_error_PANEL, 'visible','on');

end

periodic_ramp_saved=struct('Sig',periodic_ramp_ans,'Initial_value',initial_value_val,'Final_value',final_value_val,'Slope',slope_val,'Period',period,'Duration',duration);

setappdata(0,'periodic_ramp_saved',periodic_ramp_saved);

p_1=num2str(initial_value_val);
p_2=num2str(final_value_val);
p_3=num2str(slope_val);
p_4=num2str(period);
p_5=num2str(duration);
prefix=['p_ramp__' p_1 '__' p_2 '__' p_3 '__' p_4 '__' p_5 '__'];
prefix=strrep(prefix,'.','_');
prefix=strrep(prefix,'-','N');
setappdata(0,'p_ramp_prefix',prefix);


function p_ramp_SAVE_Callback(hObject, eventdata, handles)

periodic_ramp_saved=getappdata(0,'periodic_ramp_saved');

name=get(handles.p_ramp_file_name,'string');
p_ramp_prefix=char(getappdata(0,'p_ramp_prefix'));
name=char(name);
name=[p_ramp_prefix name];

save(strcat('G:\Users\Ali_R\GUI_draft\saved\periodic_ramp\',name),'periodic_ramp_saved');

set(handles.saved_PANEL,'visible','on');
set(handles.text51,'string',strcat('G:\Users\Ali_R\GUI_draft\saved\periodic_ramp\',name));



function p_ramp_duration_Callback(hObject, eventdata, handles)
% hObject    handle to p_ramp_duration (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of p_ramp_duration as text
%        str2double(get(hObject,'String')) returns contents of p_ramp_duration as a double


% --- Executes during object creation, after setting all properties.
function p_ramp_duration_CreateFcn(hObject, eventdata, handles)
% hObject    handle to p_ramp_duration (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function p_ramp_cycle_period_Callback(hObject, eventdata, handles)
% hObject    handle to p_ramp_cycle_period (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of p_ramp_cycle_period as text
%        str2double(get(hObject,'String')) returns contents of p_ramp_cycle_period as a double


% --- Executes during object creation, after setting all properties.
function p_ramp_cycle_period_CreateFcn(hObject, eventdata, handles)
% hObject    handle to p_ramp_cycle_period (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function p_ramp_initial_value_Callback(hObject, eventdata, handles)
% hObject    handle to p_ramp_initial_value (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of p_ramp_initial_value as text
%        str2double(get(hObject,'String')) returns contents of p_ramp_initial_value as a double


% --- Executes during object creation, after setting all properties.
function p_ramp_initial_value_CreateFcn(hObject, eventdata, handles)
% hObject    handle to p_ramp_initial_value (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function p_ramp_final_value_Callback(hObject, eventdata, handles)
% hObject    handle to p_ramp_final_value (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of p_ramp_final_value as text
%        str2double(get(hObject,'String')) returns contents of p_ramp_final_value as a double


% --- Executes during object creation, after setting all properties.
function p_ramp_final_value_CreateFcn(hObject, eventdata, handles)
% hObject    handle to p_ramp_final_value (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function p_ramp_slope_Callback(hObject, eventdata, handles)
% hObject    handle to p_ramp_slope (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of p_ramp_slope as text
%        str2double(get(hObject,'String')) returns contents of p_ramp_slope as a double


% --- Executes during object creation, after setting all properties.
function p_ramp_slope_CreateFcn(hObject, eventdata, handles)
% hObject    handle to p_ramp_slope (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton17.
function pushbutton17_Callback(hObject, eventdata, handles)
set(handles.p_ramp_error_PANEL, 'visible','off');



function edit27_Callback(hObject, eventdata, handles)
% hObject    handle to edit27 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit27 as text
%        str2double(get(hObject,'String')) returns contents of edit27 as a double


% --- Executes during object creation, after setting all properties.
function edit27_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit27 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton18.
function pushbutton18_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton18 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in piecewise_TAB.
function piecewise_TAB_Callback(hObject, eventdata, handles)

set(handles.sin_PANEL, 'visible','off');
set(handles.ramp_PANEL, 'visible','off');
set(handles.walk_PANEL, 'visible','off');
set(handles.p_ramp_PANEL, 'visible','off');
set(handles.piecewise_PANEL, 'visible','on');
set(handles.r_ramp_PANEL, 'visible','off');



% --- Executes on button press in piecewise_GENERATE.
function piecewise_GENERATE_Callback(hObject, eventdata, handles)
max_pf=str2double(get(handles.piecewise_min_angle,'string'));
max_df=str2double(get(handles.piecewise_max_angle,'string'));
min_hold_duration=str2double(get(handles.piecewise_min_hold_duration,'string'));
max_hold_duration=str2double(get(handles.piecewise_max_hold_duration,'string'));
duration=str2double(get(handles.piecewise_duration,'string'));

piecewise_sig=gen_piecewise_const(max_pf,max_df,min_hold_duration,max_hold_duration,duration);

piecewise_sig_amp=piecewise_sig(2,:);
piecewise_sig_time=piecewise_sig(1,:);

plot(piecewise_sig_time,piecewise_sig_amp,'Linewidth',2,'Parent',handles.piecewise_AXES);
xlabel('Time (s)','Parent',handles.piecewise_AXES);
ylabel('Position (rad)','Parent',handles.piecewise_AXES);
grid(handles.piecewise_AXES, 'on');
piecewise_saved=struct('Sig',piecewise_sig,'max_pf',max_pf,'max_df',max_df,'min_hold_duration',min_hold_duration,'max_hold_duration',max_hold_duration,'Duration',duration);

setappdata(0,'piecewise_saved',piecewise_saved);

p_1=num2str(max_pf);
p_2=num2str(max_df);
p_3=num2str(min_hold_duration);
p_4=num2str(max_hold_duration);
p_5=num2str(duration);
prefix=['piecewise__' p_1 '__' p_2 '__' p_3 '__' p_4 '__' p_5 '__'];
prefix=strrep(prefix,'.','_');
prefix=strrep(prefix,'-','N');

setappdata(0,'piecewise_prefix',prefix);



% --- Executes on button press in piecewise_SAVE.
function piecewise_SAVE_Callback(hObject, eventdata, handles)

piecewise_saved=getappdata(0,'piecewise_saved');


name=get(handles.piecewise_file_name,'string');
piecewise_prefix=char(getappdata(0,'piecewise_prefix'));
name=char(name);
name=[piecewise_prefix name];

save(strcat('G:\Users\Ali_R\GUI_draft\saved\piecewise_const\',name),'piecewise_saved');

set(handles.saved_PANEL,'visible','on');
set(handles.text51,'string',strcat('G:\Users\Ali_R\GUI_draft\saved\piecewise_const\',name));


function piecewise_max_hold_duration_Callback(hObject, eventdata, handles)
% hObject    handle to piecewise_max_hold_duration (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of piecewise_max_hold_duration as text
%        str2double(get(hObject,'String')) returns contents of piecewise_max_hold_duration as a double


% --- Executes during object creation, after setting all properties.
function piecewise_max_hold_duration_CreateFcn(hObject, eventdata, handles)
% hObject    handle to piecewise_max_hold_duration (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function piecewise_duration_Callback(hObject, eventdata, handles)
% hObject    handle to piecewise_duration (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of piecewise_duration as text
%        str2double(get(hObject,'String')) returns contents of piecewise_duration as a double


% --- Executes during object creation, after setting all properties.
function piecewise_duration_CreateFcn(hObject, eventdata, handles)
% hObject    handle to piecewise_duration (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function piecewise_min_angle_Callback(hObject, eventdata, handles)
% hObject    handle to piecewise_min_angle (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of piecewise_min_angle as text
%        str2double(get(hObject,'String')) returns contents of piecewise_min_angle as a double


% --- Executes during object creation, after setting all properties.
function piecewise_min_angle_CreateFcn(hObject, eventdata, handles)
% hObject    handle to piecewise_min_angle (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function piecewise_max_angle_Callback(hObject, eventdata, handles)
% hObject    handle to piecewise_max_angle (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of piecewise_max_angle as text
%        str2double(get(hObject,'String')) returns contents of piecewise_max_angle as a double


% --- Executes during object creation, after setting all properties.
function piecewise_max_angle_CreateFcn(hObject, eventdata, handles)
% hObject    handle to piecewise_max_angle (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function piecewise_min_hold_duration_Callback(hObject, eventdata, handles)
% hObject    handle to piecewise_min_hold_duration (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of piecewise_min_hold_duration as text
%        str2double(get(hObject,'String')) returns contents of piecewise_min_hold_duration as a double


% --- Executes during object creation, after setting all properties.
function piecewise_min_hold_duration_CreateFcn(hObject, eventdata, handles)
% hObject    handle to piecewise_min_hold_duration (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function piecewise_file_name_Callback(hObject, eventdata, handles)
% hObject    handle to piecewise_file_name (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of piecewise_file_name as text
%        str2double(get(hObject,'String')) returns contents of piecewise_file_name as a double


% --- Executes during object creation, after setting all properties.
function piecewise_file_name_CreateFcn(hObject, eventdata, handles)
% hObject    handle to piecewise_file_name (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in saved_OK.
function saved_OK_Callback(hObject, eventdata, handles)

set(handles.saved_PANEL,'visible','off');



function sin_file_name_Callback(hObject, eventdata, handles)
% hObject    handle to sin_file_name (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of sin_file_name as text
%        str2double(get(hObject,'String')) returns contents of sin_file_name as a double


% --- Executes during object creation, after setting all properties.
function sin_file_name_CreateFcn(hObject, eventdata, handles)
% hObject    handle to sin_file_name (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function p_ramp_file_name_Callback(hObject, eventdata, handles)
% hObject    handle to p_ramp_file_name (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of p_ramp_file_name as text
%        str2double(get(hObject,'String')) returns contents of p_ramp_file_name as a double


% --- Executes during object creation, after setting all properties.
function p_ramp_file_name_CreateFcn(hObject, eventdata, handles)
% hObject    handle to p_ramp_file_name (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function walk_file_name_Callback(hObject, eventdata, handles)
% hObject    handle to walk_file_name (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of walk_file_name as text
%        str2double(get(hObject,'String')) returns contents of walk_file_name as a double


% --- Executes during object creation, after setting all properties.
function walk_file_name_CreateFcn(hObject, eventdata, handles)
% hObject    handle to walk_file_name (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function ramp_file_name_Callback(hObject, eventdata, handles)
% hObject    handle to ramp_file_name (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of ramp_file_name as text
%        str2double(get(hObject,'String')) returns contents of ramp_file_name as a double


% --- Executes during object creation, after setting all properties.
function ramp_file_name_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ramp_file_name (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in r_ramp_TAB.
function r_ramp_TAB_Callback(hObject, eventdata, handles)

set(handles.sin_PANEL, 'visible','off');
set(handles.ramp_PANEL, 'visible','off');
set(handles.walk_PANEL, 'visible','off');
set(handles.p_ramp_PANEL, 'visible','off');
set(handles.piecewise_PANEL, 'visible','off');
set(handles.r_ramp_PANEL, 'visible','on');


% --- Executes on button press in r_ramp_GENERATE.
function r_ramp_GENERATE_Callback(hObject, eventdata, handles)

initial_value=str2double(get(handles.r_ramp_initial_value,'string'));
final_value=str2double(get(handles.r_ramp_final_value,'string'));
slope=str2double(get(handles.r_ramp_slope,'string'));

if initial_value>final_value
    if slope>0
      slope=slope.*-1;
    end
end


if final_value>initial_value
    if slope<0
      slope=slope.*-1;
    end
end

duration=str2double(get(handles.r_ramp_duration,'string'));
min_delay=str2double(get(handles.r_ramp_min_delay,'string'));
max_delay=str2double(get(handles.r_ramp_max_delay,'string'));

random_delay_ramp_and_hold=gen_random_delay_ramp(initial_value,final_value,slope,min_delay,max_delay,duration);
size(random_delay_ramp_and_hold)
random_delay_ramp_amp=random_delay_ramp_and_hold(2,:);
random_delay_ramp_time=random_delay_ramp_and_hold(1,:);

plot(random_delay_ramp_time,random_delay_ramp_amp,'Linewidth',2,'Parent',handles.r_ramp_AXES);
grid(handles.r_ramp_AXES, 'on');
xlabel('Time (s)','Parent',handles.r_ramp_AXES);
ylabel('Position (rad)','Parent',handles.r_ramp_AXES);
random_delay_ramp_saved=struct('Sig',random_delay_ramp_and_hold,'Initial_value',initial_value,'Final_value',final_value,'Slope',slope,'Min_delay',min_delay,'Max_delay',max_delay,'Duration',duration);

setappdata(0,'random_delay_ramp_saved',random_delay_ramp_saved);

p_1=num2str(initial_value);
p_2=num2str(final_value);
p_3=num2str(slope);
p_4=num2str(min_delay);
p_5=num2str(max_delay);
p_6=num2str(duration);

prefix=['r_ramp__' p_1 '__' p_2 '__' p_3 '__' p_4 '__' p_5 '__'];
prefix=strrep(prefix,'.','_');
prefix=strrep(prefix,'-','N');
setappdata(0,'r_ramp_prefix',prefix);

% --- Executes on button press in r_ramp_SAVE.
function r_ramp_SAVE_Callback(hObject, eventdata, handles)

random_delay_ramp=getappdata(0,'random_delay_ramp_saved');

name=get(handles.file_name_dramp,'string');

r_ramp_prefix=char(getappdata(0,'r_ramp_prefix'));
name=char(name);
name=[r_ramp_prefix name];

save(strcat('G:\Users\Ali_R\GUI_draft\saved\random_delay_ramp_and_hold\',name),'random_delay_ramp');

set(handles.saved_PANEL,'visible','on');
set(handles.text51,'string',strcat('G:\Users\Ali_R\GUI_draft\saved\random_delay_ramp_and_hold\',name));



function r_ramp_duration_Callback(hObject, eventdata, handles)
% hObject    handle to r_ramp_duration (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of r_ramp_duration as text
%        str2double(get(hObject,'String')) returns contents of r_ramp_duration as a double


% --- Executes during object creation, after setting all properties.
function r_ramp_duration_CreateFcn(hObject, eventdata, handles)
% hObject    handle to r_ramp_duration (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit46_Callback(hObject, eventdata, handles)
% hObject    handle to edit46 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit46 as text
%        str2double(get(hObject,'String')) returns contents of edit46 as a double


% --- Executes during object creation, after setting all properties.
function edit46_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit46 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function r_ramp_initial_value_Callback(hObject, eventdata, handles)
% hObject    handle to r_ramp_initial_value (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of r_ramp_initial_value as text
%        str2double(get(hObject,'String')) returns contents of r_ramp_initial_value as a double


% --- Executes during object creation, after setting all properties.
function r_ramp_initial_value_CreateFcn(hObject, eventdata, handles)
% hObject    handle to r_ramp_initial_value (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function r_ramp_final_value_Callback(hObject, eventdata, handles)
% hObject    handle to r_ramp_final_value (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of r_ramp_final_value as text
%        str2double(get(hObject,'String')) returns contents of r_ramp_final_value as a double


% --- Executes during object creation, after setting all properties.
function r_ramp_final_value_CreateFcn(hObject, eventdata, handles)
% hObject    handle to r_ramp_final_value (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function r_ramp_slope_Callback(hObject, eventdata, handles)
% hObject    handle to r_ramp_slope (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of r_ramp_slope as text
%        str2double(get(hObject,'String')) returns contents of r_ramp_slope as a double


% --- Executes during object creation, after setting all properties.
function r_ramp_slope_CreateFcn(hObject, eventdata, handles)
% hObject    handle to r_ramp_slope (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function file_name_dramp_Callback(hObject, eventdata, handles)
% hObject    handle to file_name_dramp (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of file_name_dramp as text
%        str2double(get(hObject,'String')) returns contents of file_name_dramp as a double


% --- Executes during object creation, after setting all properties.
function file_name_dramp_CreateFcn(hObject, eventdata, handles)
% hObject    handle to file_name_dramp (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function r_ramp_min_delay_Callback(hObject, eventdata, handles)
% hObject    handle to r_ramp_min_delay (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of r_ramp_min_delay as text
%        str2double(get(hObject,'String')) returns contents of r_ramp_min_delay as a double


% --- Executes during object creation, after setting all properties.
function r_ramp_min_delay_CreateFcn(hObject, eventdata, handles)
% hObject    handle to r_ramp_min_delay (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function r_ramp_max_delay_Callback(hObject, eventdata, handles)
% hObject    handle to r_ramp_max_delay (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of r_ramp_max_delay as text
%        str2double(get(hObject,'String')) returns contents of r_ramp_max_delay as a double


% --- Executes during object creation, after setting all properties.
function r_ramp_max_delay_CreateFcn(hObject, eventdata, handles)
% hObject    handle to r_ramp_max_delay (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes during object creation, after setting all properties.
function p_ramp_AXES_CreateFcn(hObject, eventdata, handles)
% hObject    handle to p_ramp_AXES (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate p_ramp_AXES


% --- Executes on button press in pushbutton29.
function pushbutton29_Callback(hObject, eventdata, handles)

set(handles.a, 'visible','off');


% --- Executes during object creation, after setting all properties.
function a_CreateFcn(hObject, eventdata, handles)
% hObject    handle to a (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes during object deletion, before destroying properties.
function a_DeleteFcn(hObject, eventdata, handles)
% hObject    handle to a (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
