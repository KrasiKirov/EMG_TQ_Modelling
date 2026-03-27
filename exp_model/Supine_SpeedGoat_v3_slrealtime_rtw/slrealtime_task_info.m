function [taskInfo, numtask, isDeploymentDiagram]=slrealtime_task_info()
   taskInfo(1).samplePeriod = 0.001;
   taskInfo(1).sampleOffset = 0.0;
   taskInfo(1).taskPrio = 15;
   taskInfo(1).taskName = 'BaseRate';
   taskInfo(1).entryPoints = {};
   taskInfo(2).samplePeriod = 0.02;
   taskInfo(2).sampleOffset = 0.0;
   taskInfo(2).taskPrio = 14;
   taskInfo(2).taskName = ['SubRate' '1'];        
   taskInfo(2).entryPoints = {};
   taskInfo(3).samplePeriod = 0.2;
   taskInfo(3).sampleOffset = 0.0;
   taskInfo(3).taskPrio = 13;
   taskInfo(3).taskName = ['SubRate' '2'];        
   taskInfo(3).entryPoints = {};
   taskInfo(4).samplePeriod = 0.5;
   taskInfo(4).sampleOffset = 0.0;
   taskInfo(4).taskPrio = 12;
   taskInfo(4).taskName = ['SubRate' '3'];        
   taskInfo(4).entryPoints = {};
   taskInfo(5).samplePeriod = 0.7;
   taskInfo(5).sampleOffset = 0.0;
   taskInfo(5).taskPrio = 11;
   taskInfo(5).taskName = ['SubRate' '4'];        
   taskInfo(5).entryPoints = {};
   taskInfo(6).samplePeriod = 1.0;
   taskInfo(6).sampleOffset = 0.0;
   taskInfo(6).taskPrio = 10;
   taskInfo(6).taskName = ['SubRate' '5'];        
   taskInfo(6).entryPoints = {};
   taskInfo(7).samplePeriod = 2.0;
   taskInfo(7).sampleOffset = 0.0;
   taskInfo(7).taskPrio = 9;
   taskInfo(7).taskName = ['SubRate' '6'];        
   taskInfo(7).entryPoints = {};
   numtask = 7;
   for i = 1:numtask
      if ( 0 == isnumeric(taskInfo(i).samplePeriod) )
         taskInfo(i).samplePeriod = evalin('base', taskInfo(i).samplePeriod);
      end
      if ( isempty(taskInfo(i).taskName) )
         taskInfo(i).taskName = ['AutoGen' i ];
      end
   end
   isDeploymentDiagram = 0;
end 
