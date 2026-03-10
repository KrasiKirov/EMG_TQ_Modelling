function [input_final]=add_transitions(input)

ls=input(:,1);
pert=input(:,2);
size(ls)
size(pert)
initial_value=ls(1);
final_value=ls(end);
start_hold=initial_value.*ones(3000,1);
end_hold=final_value.*ones(3000,1);

if initial_value~=0 && final_value~=0

    
    
    %%%%%%%%%%%%%%%
        if initial_value>0
            transition_slope=0.05
        else
            transition_slope=-0.05
        end
        
start_transition=gen_ramp_and_hold(0,initial_value,transition_slope,0,0);

start_transition=[start_transition(2,:)' ; start_hold ];




        if final_value>0
            transition_slope=-0.05
        else
            transition_slope=0.05
        end

end_transition=gen_ramp_and_hold(final_value,0,transition_slope,0,0);

end_transition=[end_hold ; end_transition(2,:)' ;zeros(30000,1)];
    %%%%%%%%%%%%%%
    
    
    
else
    end_transition=zeros(20000,1);
    start_transition=zeros(3000,1);
    
    
end




pert_start_transition=zeros(1,size(start_transition,1))';
pert_end_transition=zeros(1,size(end_transition,1))';


ls_final=[start_transition;ls;end_transition];
pert_final=[pert_start_transition;pert;pert_end_transition];
input_final=[ls_final pert_final];