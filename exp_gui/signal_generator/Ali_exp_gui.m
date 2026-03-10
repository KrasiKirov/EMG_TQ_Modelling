function varargout = Ali_exp_gui(varargin)
% ALI_EXP_GUI MATLAB code for Ali_exp_gui.fig
%      ALI_EXP_GUI, by itself, creates a new ALI_EXP_GUI or raises the existing
%      singleton*.
%
%      H = ALI_EXP_GUI returns the handle to a new ALI_EXP_GUI or the handle to
%      the existing singleton*.
%
%      ALI_EXP_GUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in ALI_EXP_GUI.M with the given input arguments.
%
%      ALI_EXP_GUI('Property','Value',...) creates a new ALI_EXP_GUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before Ali_exp_gui_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to Ali_exp_gui_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help Ali_exp_gui

% Last Modified by GUIDE v2.5 28-Nov-2019 05:10:31

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @Ali_exp_gui_OpeningFcn, ...
                   'gui_OutputFcn',  @Ali_exp_gui_OutputFcn, ...
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


% --- Executes just before Ali_exp_gui is made visible.
function Ali_exp_gui_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to Ali_exp_gui (see VARARGIN)

% Choose default command line output for Ali_exp_gui
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes Ali_exp_gui wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = Ali_exp_gui_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;

% --- Executes on button press in experiment_design_TAB.
function experiment_design_TAB_Callback(hObject, eventdata, handles)

set(handles.experiment_design_PANEL,'visible','on');
set(handles.experiment_control_PANEL,'visible','off');


% --- Executes on button press in experiment_control_TAB.
function experiment_control_TAB_Callback(hObject, eventdata, handles)

set(handles.experiment_design_PANEL,'visible','off');
set(handles.experiment_control_PANEL,'visible','on');
set(handles.uipanel13,'visible','on');

% --- Executes on selection change in experiment_design_ls_POP.
function experiment_design_ls_POP_Callback(hObject, eventdata, handles)

allItems = get(handles.experiment_design_ls_POP,'string');
selectedIndex = get(handles.experiment_design_ls_POP,'Value');
selectedItem = allItems{selectedIndex};


switch selectedItem
    
    case 'Ramp & hold - random delay'
 
        
% after a signal is selected the "DONE" button Is deactivated this is because
% when a new signal is chosen it has to go through the safety checks which 
% is performed in the done call back function so this prohibits the 
% application to save or use signals which are not checked by the done 
% callback function.        
     
        set(handles.input_design_DONE,'Enable','off');
        
        
        [dramp_FileName,dramp_Path] = uigetfile('G:\Users\Ali_R\GUI_draft\saved\random_delay_ramp_and_hold');
        dramp_struct=importdata([dramp_Path dramp_FileName]);
        dramp_initial_value=dramp_struct.Initial_value;
        dramp_final_value=dramp_struct.Final_value;
        dramp_slope=dramp_struct.Slope;
        min_delay=dramp_struct.Min_delay;
        max_delay=dramp_struct.Max_delay;
        dramp_sig=dramp_struct.Sig;
        dramp_duration=dramp_struct.Duration;
        
        setappdata(0,'selected_large_signal',dramp_struct);
        setappdata(0,'ls_name',dramp_FileName);
        
        set(handles.dc_PANEL,'visible','off')
        set(handles.sin_parameters_PANEL,'visible','off')
        set(handles.ramp_parameters_PANEL,'visible','off')
        set(handles.walk_parameters_PANEL,'visible','off')
        set(handles.p_ramp_parameters_PANEL,'visible','off')
        set(handles.piecewise_parameters_PANEL,'visible','off')
             set(handles.r_ramp_parameters_PANEL,'visible','on');
        
        set(handles.text111,'string',dramp_FileName);
        set(handles.t_1,'string',dramp_initial_value);
        set(handles.t_2,'string',dramp_final_value);
        set(handles.t_3,'string',dramp_slope);
        set(handles.t_4,'string',min_delay);
        set(handles.t_5,'string',max_delay);
        set(handles.t_6,'string',dramp_duration);
        
       plot(dramp_sig(1,:),dramp_sig(2,:),'Linewidth',2,'Parent',handles.experiment_design_ls_AXES);
       xlabel('Time (s)','Parent',handles.experiment_design_ls_AXES);
       ylabel('Amplitude (rad)','Parent',handles.experiment_design_ls_AXES);
       grid(handles.experiment_design_ls_AXES, 'on');
    
    case 'Piecewise constant'
        
        set(handles.input_design_DONE,'Enable','off');
        
            fs=1000;
            
        [piecewise_const_FileName,piecewise_const_Path] = uigetfile('G:\Users\Ali_R\GUI_draft\saved\piecewise_const')
        piecewise_const_struct=importdata([piecewise_const_Path piecewise_const_FileName]);
        piecewise_const_max_pf= piecewise_const_struct.max_pf;
        piecewise_const_max_df= piecewise_const_struct.max_df;
        piecewise_const_min_hold_duration= piecewise_const_struct.min_hold_duration;
        piecewise_const_max_hold_duration= piecewise_const_struct.max_hold_duration;
        piecewise_const_Duration= piecewise_const_struct.Duration;
        
        
    
   
        piecewise_const_sig= piecewise_const_struct.Sig;
        
        setappdata(0,'selected_large_signal', piecewise_const_struct);
        
        setappdata(0,'ls_name',piecewise_const_FileName);
        
        set(handles.dc_PANEL,'visible','off')
        set(handles.sin_parameters_PANEL,'visible','off')
        set(handles.ramp_parameters_PANEL,'visible','off')
        set(handles.walk_parameters_PANEL,'visible','off')
        set(handles.p_ramp_parameters_PANEL,'visible','off')
             set(handles.piecewise_parameters_PANEL,'visible','on')
        set(handles.r_ramp_parameters_PANEL,'visible','off');
       
        
        set(handles.text100,'string',piecewise_const_FileName);
        set(handles.text98,'string',piecewise_const_max_df);
        set(handles.text99,'string',piecewise_const_max_pf);
        set(handles.text97,'string',piecewise_const_min_hold_duration);
        set(handles.text96,'string',piecewise_const_max_hold_duration);
        set(handles.text102,'string',piecewise_const_Duration);
        
        plot(piecewise_const_sig(1,:),piecewise_const_sig(2,:),'Linewidth',2,'Parent',handles.experiment_design_ls_AXES);
        xlabel('Time (s)','Parent',handles.experiment_design_ls_AXES);
        ylabel('Amplitude (rad)','Parent',handles.experiment_design_ls_AXES);
        grid(handles.experiment_design_ls_AXES, 'on');
        
        
        
    case 'Ramp & hold - periodic'
        
        set(handles.input_design_DONE,'Enable','off');
        
            fs=1000;
            
        [periodic_ramp_FileName,periodic_ramp_Path] = uigetfile('G:\Users\Ali_R\GUI_draft\saved\periodic_ramp')
        periodic_ramp_struct=importdata([periodic_ramp_Path periodic_ramp_FileName]);
        periodic_ramp_period=periodic_ramp_struct.Period;
        periodic_ramp_Initial_value=periodic_ramp_struct.Initial_value;
        periodic_ramp_Final_value=periodic_ramp_struct.Final_value;
        periodic_ramp_Slope=periodic_ramp_struct.Slope;
        
        
    
        
        periodic_ramp_sig=periodic_ramp_struct.Sig;
        
        periodic_ramp_Duration=size(periodic_ramp_sig(2,:),2)/fs;
        
        setappdata(0,'selected_large_signal',periodic_ramp_struct);
        
        setappdata(0,'ls_name',periodic_ramp_FileName);
        
        set(handles.dc_PANEL,'visible','off')
        set(handles.sin_parameters_PANEL,'visible','off')
        set(handles.ramp_parameters_PANEL,'visible','off')
        set(handles.walk_parameters_PANEL,'visible','off')
              set(handles.p_ramp_parameters_PANEL,'visible','on')
        set(handles.piecewise_parameters_PANEL,'visible','off')
        set(handles.r_ramp_parameters_PANEL,'visible','off');
      
        set(handles.text72,'string',periodic_ramp_FileName);
        set(handles.text70,'string',periodic_ramp_Initial_value);
        set(handles.text71,'string',periodic_ramp_Final_value);
        set(handles.text73,'string',periodic_ramp_Slope);
        set(handles.text74,'string',periodic_ramp_period);
        set(handles.text75,'string',periodic_ramp_Duration);
        
        
        plot(periodic_ramp_sig(1,:),periodic_ramp_sig(2,:),'Linewidth',2,'Parent',handles.experiment_design_ls_AXES);
        xlabel('Time (s)','Parent',handles.experiment_design_ls_AXES);
        ylabel('Amplitude (rad)','Parent',handles.experiment_design_ls_AXES);
        grid(handles.experiment_design_ls_AXES, 'on');
 
    
    
    case 'Sinusoidal'
        
        set(handles.input_design_DONE,'Enable','off');
        
        [sin_FileName,sin_Path] = uigetfile('G:\Users\Ali_R\GUI_draft\saved\sin')
        sin_struct=importdata([sin_Path sin_FileName]);
        sin_freq=sin_struct.sin_freq;
        sin_amp=sin_struct.sin_amp;
        sin_duration=sin_struct.Duration;
        
        sin_sig=sin_struct.Sig;
        
        setappdata(0,'selected_large_signal',sin_struct);
        setappdata(0,'ls_name',sin_FileName);
        
        set(handles.dc_PANEL,'visible','off')
             set(handles.sin_parameters_PANEL,'visible','on')
        set(handles.ramp_parameters_PANEL,'visible','off')
        set(handles.walk_parameters_PANEL,'visible','off')
        set(handles.piecewise_parameters_PANEL,'visible','off')
        set(handles.p_ramp_parameters_PANEL,'visible','off')  
        set(handles.r_ramp_parameters_PANEL,'visible','off');
        
        set(handles.text52,'string',sin_FileName);
        set(handles.text51,'string',sin_amp);
        set(handles.text50,'string',sin_freq);
        set(handles.text53,'string',sin_duration);
        
        plot(sin_sig(1,:),sin_sig(2,:),'Linewidth',2,'Parent',handles.experiment_design_ls_AXES);
        
        set(handles.experiment_design_ls_AXES,'XLim',[0,10]);
        xlabel('Time (s)','Parent',handles.experiment_design_ls_AXES);
        ylabel('Amplitude (rad)','Parent',handles.experiment_design_ls_AXES);
        grid(handles.experiment_design_ls_AXES, 'on');
        
    case 'Ramp & hold'
       
        set(handles.input_design_DONE,'Enable','off');
        
        [ramp_FileName,ramp_Path] = uigetfile('G:\Users\Ali_R\GUI_draft\saved\ramp');
        ramp_struct=importdata([ramp_Path ramp_FileName]);
        ramp_initial_value=ramp_struct.Initial_value;
        ramp_final_value=ramp_struct.Final_value;
        ramp_slope=ramp_struct.Slope;
        ramp_initial_hold=ramp_struct.Initial_hold;
        ramp_final_hold=ramp_struct.Final_hold;
        ramp_sig=ramp_struct.Sig;
        ramp_duration=ramp_struct.Duration;
        
        setappdata(0,'selected_large_signal',ramp_struct);
        setappdata(0,'ls_name',ramp_FileName);
        
        set(handles.dc_PANEL,'visible','off')
        set(handles.sin_parameters_PANEL,'visible','off')
                set(handles.ramp_parameters_PANEL,'visible','on')
        set(handles.walk_parameters_PANEL,'visible','off')
        set(handles.p_ramp_parameters_PANEL,'visible','off')
        set(handles.piecewise_parameters_PANEL,'visible','off')
        set(handles.r_ramp_parameters_PANEL,'visible','off');
        
        set(handles.text36,'string',ramp_FileName);
        set(handles.text35,'string',ramp_initial_value);
        set(handles.text34,'string',ramp_final_value);
        set(handles.text33,'string',ramp_slope);
        set(handles.text32,'string',ramp_initial_hold);
        set(handles.text10,'string',ramp_final_hold);
        set(handles.text87,'string',ramp_duration);
        
       plot(ramp_sig(1,:),ramp_sig(2,:),'Linewidth',2,'Parent',handles.experiment_design_ls_AXES);
       xlabel('Time (s)','Parent',handles.experiment_design_ls_AXES);
       ylabel('Amplitude (rad)','Parent',handles.experiment_design_ls_AXES);
       grid(handles.experiment_design_ls_AXES, 'on');
        
    case 'Walk'
        
        set(handles.input_design_DONE,'Enable','off');
        
        disp('walk')
        
        [walk_FileName,walk_Path] = uigetfile('G:\Users\Ali_R\GUI_draft\saved\walk')
        walk_struct=importdata([walk_Path walk_FileName]);
        walk_ROM_low=walk_struct.walk_ROM_low;
        walk_ROM_high=walk_struct.walk_ROM_high;
        walk_freq=walk_struct.walk_period;
        walk_duration=walk_struct.Duration;
        walk_sig=walk_struct.Sig; 
        
        setappdata(0,'selected_large_signal',walk_struct);
        setappdata(0,'ls_name',walk_FileName);
         
        set(handles.dc_PANEL,'visible','off')
        set(handles.sin_parameters_PANEL,'visible','off')
        set(handles.ramp_parameters_PANEL,'visible','off')
                set(handles.walk_parameters_PANEL,'visible','on')
        set(handles.p_ramp_parameters_PANEL,'visible','off')
        set(handles.p_ramp_parameters_PANEL,'visible','off')
        set(handles.piecewise_parameters_PANEL,'visible','off')
         
        set(handles.text44,'string',walk_FileName);
        set(handles.text43,'string',walk_ROM_high);
        set(handles.text42,'string',walk_ROM_low);
        set(handles.text41,'string',walk_freq);
        set(handles.text40,'string',walk_duration);
        set(handles.r_ramp_parameters_PANEL,'visible','off');
        
        plot(walk_sig(1,:),walk_sig(2,:),'Linewidth',2,'Parent',handles.experiment_design_ls_AXES);
        set(handles.experiment_design_ls_AXES,'XLim',[0,10]);
        
        xlabel('Time (s)','Parent',handles.experiment_design_ls_AXES);
        ylabel('Amplitude (rad)','Parent',handles.experiment_design_ls_AXES);
        grid(handles.experiment_design_ls_AXES, 'on');
        
     case 'None'
         
         set(handles.input_design_DONE,'Enable','off');
         % actual selection of the DC signal is done via dc_PANEL
         
              set(handles.dc_PANEL,'visible','on')
        set(handles.p_ramp_parameters_PANEL,'visible','off')
        set(handles.sin_parameters_PANEL,'visible','off')
        set(handles.ramp_parameters_PANEL,'visible','off')
        set(handles.walk_parameters_PANEL,'visible','off')
        set(handles.p_ramp_parameters_PANEL,'visible','off')
        set(handles.piecewise_parameters_PANEL,'visible','off')
        
       
        

         
        
     case 'New'
         
        set(handles.input_design_DONE,'Enable','off');
        disp('new')
        run('Signal_generator.m');
        
        
end




% --- Executes during object creation, after setting all properties.
function experiment_design_ls_POP_CreateFcn(hObject, eventdata, handles)
% hObject    handle to experiment_design_ls_POP (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit1_Callback(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit1 as text
%        str2double(get(hObject,'String')) returns contents of edit1 as a double


% --- Executes during object creation, after setting all properties.
function edit1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit3_Callback(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit3 as text
%        str2double(get(hObject,'String')) returns contents of edit3 as a double


% --- Executes during object creation, after setting all properties.
function edit3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in experiment_design_pert_POP.
function experiment_design_pert_POP_Callback(hObject, eventdata, handles)

allItems = get(handles.experiment_design_pert_POP,'string');
selectedIndex = get(handles.experiment_design_pert_POP,'Value');
selectedItem = allItems{selectedIndex};

 switch selectedItem
     
       case 'None'
         set(handles.input_design_DONE,'Enable','off');
         
         
         fs=1000;
         none_duration=200;
         none_amp=zeros(1,200000);
         none_time=(0:1/fs:none_duration-1/fs);
         none_sig=[none_amp];
         none_struct=struct('Sig',none_sig,'Duration',none_duration);
         
         setappdata(0,'selected_perturbation',none_struct);
         setappdata(0,'pert_name','None');
        
        set(handles.pralds_parameters_PANEL,'visible','off')
        set(handles.prbs_parameters_PANEL,'visible','off')
        
        plot(none_amp,'Linewidth',2,'Parent',handles.experiment_design_pert_AXES);
        grid(handles.experiment_design_pert_AXES, 'on');

        xlabel('Time (s)','Parent',handles.experiment_design_pert_AXES);
        ylabel('Amplitude (rad)','Parent',handles.experiment_design_pert_AXES);
        
     case 'New'
         
         set(handles.input_design_DONE,'Enable','off');
         run('Perturbation_generator.m')
         
     case 'PRBS'     
         
         set(handles.input_design_DONE,'Enable','off');
                     
         [prbs_FileName,prbs_Path] = uigetfile('G:\Users\Ali_R\GUI_draft\saved\prbs');
          prbs_struct=importdata([prbs_Path prbs_FileName]);
          Amplitude=prbs_struct.Amplitude;
          prbs_duration=prbs_struct.Duration;
          Pulse_width=prbs_struct.Pulse_width;
          prbs_sig=prbs_struct.Sig;
         
          setappdata(0,'selected_perturbation',prbs_struct);
          setappdata(0,'pert_name',prbs_FileName);
          
          
        set(handles.pralds_parameters_PANEL,'visible','off')
        set(handles.prbs_parameters_PANEL,'visible','on')
          
         set(handles.text58,'string',prbs_FileName);
         set(handles.text57,'string',Amplitude/2);
         set(handles.text56,'string',Pulse_width);
         set(handles.text59,'string',prbs_duration);

        
        plot(prbs_sig,'Linewidth',2,'Parent',handles.experiment_design_pert_AXES);
        xlabel('Time (s)','Parent',handles.experiment_design_pert_AXES);
        ylabel('Amplitude (rad)','Parent',handles.experiment_design_pert_AXES);
        set(handles.experiment_design_pert_AXES,'XLim',[0,10000]);
        grid(handles.experiment_design_pert_AXES, 'on');
        
 case 'PRALDS'     
         
          set(handles.input_design_DONE,'Enable','off');
          
         [pralds_FileName,pralds_Path] = uigetfile('G:\Users\Ali_R\GUI_draft\saved\pralds');
          pralds_struct=importdata([pralds_Path pralds_FileName]);
          Amplitude=pralds_struct.Amp;
          pralds_duration=pralds_struct.Duration;
          Min_duration=pralds_struct.Min_duration;
          Max_duration=pralds_struct.Max_duration;
          pralds_sig=pralds_struct.Sig;
          time=pralds_struct.Time;
          
          
          setappdata(0,'selected_perturbation',pralds_struct);
          setappdata(0,'pert_name',pralds_FileName);
          
        set(handles.pralds_parameters_PANEL,'visible','on')
        set(handles.prbs_parameters_PANEL,'visible','off')
          
         set(handles.text83,'string',pralds_FileName);
         set(handles.pralds_amp,'string',Amplitude);
         set(handles.min_duration,'string',Min_duration);
         set(handles.max_duration,'string',Max_duration);
         
        
        plot(time,pralds_sig,'Linewidth',2,'Parent',handles.experiment_design_pert_AXES); 
        
        xlabel('Time (s)','Parent',handles.experiment_design_pert_AXES);
        ylabel('Amplitude (rad)','Parent',handles.experiment_design_pert_AXES);
        set(handles.experiment_design_pert_AXES,'XLim',[0,10]);
        grid(handles.experiment_design_pert_AXES, 'on');
        
        
 end


% --- Executes during object creation, after setting all properties.
function experiment_design_pert_POP_CreateFcn(hObject, eventdata, handles)
% hObject    handle to experiment_design_pert_POP (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



function edit4_Callback(hObject, eventdata, handles)
% hObject    handle to edit4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit4 as text
%        str2double(get(hObject,'String')) returns contents of edit4 as a double


% --- Executes during object creation, after setting all properties.
function edit4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton5.
function pushbutton5_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in input_design_continue.
function input_design_continue_Callback(hObject, eventdata, handles)
fs=1000;

large_signal=getappdata(0,'selected_large_signal');
perturbations=getappdata(0,'selected_perturbation');


ls_duration=large_signal.Duration;
p_duration=perturbations.Duration;

offset=str2double(get(handles.offset,'string'));
large_signal.Sig(2,:)=large_signal.Sig(2,:)+offset;
trimmed_perturbation=zeros(1,1000*ls_duration);
amp=large_signal.Sig;
pert_amp=perturbations.Sig;
amp=amp(2,:);

max_pert_val=max(max(pert_amp))
min_pert_val=min(min(pert_amp))

allowed_pert_amplitude=0.1;

set(handles.dialog_PANEL,'visible','off');

if allowed_pert_amplitude > max_pert_val & min_pert_val > allowed_pert_amplitude*(-1)
    set(handles.input_design_DONE,'Enable','on')
else
    
    set(handles.pert_amp_warning_PANEL,'visible','on');
    set(handles.input_design_DONE,'Enable','off');
    
end




for n=(1:1:(fs*(ls_duration)-1/fs))
    trimmed_perturbation(n)=pert_amp(n);
end

 amp_size=size(amp)
 pert_size=size(trimmed_perturbation)
 
final_sig= trimmed_perturbation+amp;
time=(0:1/fs:ls_duration-1/fs);

plot(time,final_sig,'Linewidth',2,'Parent',handles.axes4);
xlabel('Time (s)','Parent',handles.axes4);
ylabel('Amplitude (rad)','Parent',handles.axes4);
grid(handles.axes4, 'on');


final_sig_struct=struct('ls',large_signal,'pert',trimmed_perturbation);

setappdata(0,'final_sig_struct',final_sig_struct);

% --- Executes on button press in stop_perturbation.
function stop_perturbation_Callback(hObject, eventdata, handles)

global TG
setparam(TG, getparamid(TG, 'position_switch','Value'),0);


function file_pos_name_Callback(hObject, eventdata, handles)
% hObject    handle to file_pos_name (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of file_pos_name as text
%        str2double(get(hObject,'String')) returns contents of file_pos_name as a double


% --- Executes during object creation, after setting all properties.
function file_pos_name_CreateFcn(hObject, eventdata, handles)
% hObject    handle to file_pos_name (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in browse_input.
function browse_input_Callback(hObject, eventdata, handles)

global file_pos_name path_pos;

[file_pos_name,path_pos] = uigetfile('G:\Users\Ali_R\GUI_draft\saved\input');
 if file_pos_name==0
     file_pos_name='ERROR';
 end
 
final_sig_struct=importdata([path_pos file_pos_name]);

setappdata(0,'input',final_sig_struct);
        
set(handles.file_pos_name,'String',file_pos_name);
set(handles.go,'Enable','off')





function pos_amp_Callback(hObject, eventdata, handles)
% hObject    handle to pos_amp (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of pos_amp as text
%        str2double(get(hObject,'String')) returns contents of pos_amp as a double


% --- Executes during object creation, after setting all properties.
function pos_amp_CreateFcn(hObject, eventdata, handles)
% hObject    handle to pos_amp (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in go.
function go_Callback(hObject, eventdata, handles)

global file_pos_name path_pos file_tq_name path_tq TG;

input=getappdata(0,'input');
input_signal_pert=(input.pert)';
input_signal_ls=(input.ls.Sig(2,:))';


input=[input_signal_ls input_signal_pert];

input=add_transitions(input);

%address_pos=strcat(path_pos,file_pos_name);
address_tq=strcat(path_tq,file_tq_name);
%load(address_pos);
load(address_tq);
% input=eval(file_pos_name(1:end-4));

% if size(input,2)> size(input,1)
%     input = input';
% end


input=input*10;

visual=eval(file_tq_name(1:end-4));
if size(visual,2)> size(visual,1)
    visual = visual';
end

length_in = size(input,1);
length_v = length(visual);
m = min(length_in,length_v);
input = [input(1:m,:) visual(1:m,:)];
% buffer_length=get(handles.buf_len,'String');
% buffer_length=str2double(buffer_length);

buffer_length=10000;
upload1(TG,input,buffer_length);


setparam(TG, getparamid(TG, 'position_switch','Value'),1);



function tq_file_name_Callback(hObject, eventdata, handles)
% hObject    handle to tq_file_name (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of tq_file_name as text
%        str2double(get(hObject,'String')) returns contents of tq_file_name as a double


% --- Executes during object creation, after setting all properties.
function tq_file_name_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tq_file_name (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function buf_len_Callback(hObject, eventdata, handles)
% hObject    handle to buf_len (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of buf_len as text
%        str2double(get(hObject,'String')) returns contents of buf_len as a double


% --- Executes during object creation, after setting all properties.
function buf_len_CreateFcn(hObject, eventdata, handles)
% hObject    handle to buf_len (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in browse_visual.
function browse_visual_Callback(hObject, eventdata, handles)

global file_tq_name path_tq;
[file_tq_name,path_tq,FilterIndex] = uigetfile('G:\Users\Ali_R\GUI_draft\monitor');
 if file_tq_name==0
     file_tq_name='ERROR';
 end
set(handles.tq_file_name,'String',file_tq_name);


% --- Executes on button press in input_design_save.
function input_design_save_Callback(hObject, eventdata, handles)

input=getappdata(0,'final_sig_struct');
ls_prefix=getappdata(0,'ls_name');
pert_prefix=getappdata(0,'pert_name');
prefix=[ls_prefix '_' pert_prefix '_'];
input_name=get(handles.input_name,'string')

name= ['input_' prefix input_name];

name=strrep(name,' ','');
name=strrep(name,'.mat','');

path=strcat('D:\Users\Ali_R\GUI_draft\saved\input\',name)
save(path,'input');

set(handles.saved_PANEL,'visible','on');
set(handles.saved_txt,'string',strcat('G:\Users\Ali_R\GUI_draft\saved\input\',name));
set(handles.dialog_PANEL,'visible','off');





function input_name_Callback(hObject, eventdata, handles)
% hObject    handle to input_name (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of input_name as text
%        str2double(get(hObject,'String')) returns contents of input_name as a double


% --- Executes during object creation, after setting all properties.
function input_name_CreateFcn(hObject, eventdata, handles)
% hObject    handle to input_name (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in input_design_use.
function input_design_use_Callback(hObject, eventdata, handles)

input=getappdata(0,'final_sig_struct');
setappdata(0,'input',input);
set(handles.experiment_design_PANEL,'visible','off');
set(handles.experiment_control_PANEL,'visible','on');
set(handles.uipanel13,'visible','off');
set(handles.go,'Enable','off');
set(handles.dialog_PANEL,'visible','off');




% --- Executes on button press in session_control_SET.
function session_control_SET_Callback(hObject, eventdata, handles)

input=getappdata(0,'input');
rom_min=str2double(get(handles.rom_min,'string'));
rom_max=str2double(get(handles.rom_max,'string'));

input_signal_ls=(input.ls.Sig(2,:))';

max_val=max(max(input_signal_ls));
min_val=min(min(input_signal_ls));


if rom_max > max_val & rom_min < min_val
    set(handles.go,'Enable','on')
else
    
    set(handles.amp_warning_PANEL,'visible','on');
    set(handles.go,'Enable','off');
    
end

    






function rom_min_Callback(hObject, eventdata, handles)
% hObject    handle to rom_min (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of rom_min as text
%        str2double(get(hObject,'String')) returns contents of rom_min as a double


% --- Executes during object creation, after setting all properties.
function rom_min_CreateFcn(hObject, eventdata, handles)
% hObject    handle to rom_min (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function rom_max_Callback(hObject, eventdata, handles)
% hObject    handle to rom_max (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of rom_max as text
%        str2double(get(hObject,'String')) returns contents of rom_max as a double


% --- Executes during object creation, after setting all properties.
function rom_max_CreateFcn(hObject, eventdata, handles)
% hObject    handle to rom_max (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in warning_panel_OK.
function warning_panel_OK_Callback(hObject, eventdata, handles)

set(handles.amp_warning_PANEL,'visible','off');


% --- Executes on button press in hello.
function hello_Callback(hObject, eventdata, handles)

str=get(handles.ahmad,'string');
str



function ahmad_Callback(hObject, eventdata, handles)
% hObject    handle to ahmad (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of ahmad as text
%        str2double(get(hObject,'String')) returns contents of ahmad as a double


% --- Executes during object creation, after setting all properties.
function ahmad_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ahmad (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes during object creation, after setting all properties.
function experiment_design_pert_AXES_CreateFcn(hObject, eventdata, handles)
% hObject    handle to experiment_design_pert_AXES (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate experiment_design_pert_AXES


% --- Executes during object deletion, before destroying properties.
function experiment_design_pert_AXES_DeleteFcn(hObject, eventdata, handles)
% hObject    handle to experiment_design_pert_AXES (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes during object deletion, before destroying properties.
function buf_len_DeleteFcn(hObject, eventdata, handles)
% hObject    handle to buf_len (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



function edit13_Callback(hObject, eventdata, handles)
% hObject    handle to buf_len (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of buf_len as text
%        str2double(get(hObject,'String')) returns contents of buf_len as a double


% --- Executes during object creation, after setting all properties.
function edit13_CreateFcn(hObject, eventdata, handles)
% hObject    handle to buf_len (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function offset_Callback(hObject, eventdata, handles)
% hObject    handle to offset (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of offset as text
%        str2double(get(hObject,'String')) returns contents of offset as a double


% --- Executes during object creation, after setting all properties.
function offset_CreateFcn(hObject, eventdata, handles)
% hObject    handle to offset (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in saved_button.
function saved_button_Callback(hObject, eventdata, handles)
% hObject    handle to saved_button (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.saved_PANEL,'visible','off');



function edit15_Callback(hObject, eventdata, handles)
% hObject    handle to edit15 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit15 as text
%        str2double(get(hObject,'String')) returns contents of edit15 as a double


% --- Executes during object creation, after setting all properties.
function edit15_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit15 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit16_Callback(hObject, eventdata, handles)
% hObject    handle to edit16 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit16 as text
%        str2double(get(hObject,'String')) returns contents of edit16 as a double


% --- Executes during object creation, after setting all properties.
function edit16_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit16 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton23.
function pushbutton23_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton23 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

fs=1000;


none_duration=str2double(get(handles.edit16,'string'));

         none_amp=zeros(1,none_duration.*fs);
         none_time=(0:1/fs:none_duration-1/fs);
         none_sig=[none_time;none_amp];
         None_struct=struct('Sig',none_sig,'Duration',none_duration);
         setappdata(0,'selected_large_signal',None_struct);
         setappdata(0,'ls_name','None');
        
        
        
        
        plot(none_sig(1,:),none_sig(2,:),'Linewidth',2,'Parent',handles.experiment_design_ls_AXES);
        xlabel('Time (s)','Parent',handles.experiment_design_ls_AXES);
        ylabel('Amplitude (rad)','Parent',handles.experiment_design_ls_AXES);
        grid(handles.experiment_design_ls_AXES, 'on');


% --- If Enable == 'on', executes on mouse press in 5 pixel border.
% --- Otherwise, executes on mouse press in 5 pixel border or over experiment_design_TAB.
function experiment_design_TAB_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to experiment_design_TAB (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



function edit17_Callback(hObject, eventdata, handles)
% hObject    handle to edit17 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit17 as text
%        str2double(get(hObject,'String')) returns contents of edit17 as a double


% --- Executes during object creation, after setting all properties.
function edit17_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit17 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton24.
function pushbutton24_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton24 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in input_design_DONE.
function input_design_DONE_Callback(hObject, eventdata, handles)
% hObject    handle to input_design_DONE (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

set(handles.dialog_PANEL,'visible','on');


function edit21_Callback(hObject, eventdata, handles)
% hObject    handle to edit21 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit21 as text
%        str2double(get(hObject,'String')) returns contents of edit21 as a double


% --- Executes during object creation, after setting all properties.
function edit21_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit21 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit20_Callback(hObject, eventdata, handles)
% hObject    handle to edit20 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit20 as text
%        str2double(get(hObject,'String')) returns contents of edit20 as a double


% --- Executes during object creation, after setting all properties.
function edit20_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit20 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pert_amp_warning_OK.
function pert_amp_warning_OK_Callback(hObject, eventdata, handles)

set(handles.pert_amp_warning_PANEL,'visible','off');
