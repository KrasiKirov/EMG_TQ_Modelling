function xcp = Supine_SpeedGoat_v3xcp

xcp.events     =  repmat(struct('id',{}, 'sampletime', {}, 'offset', {}), getNumEvents, 1 );
xcp.parameters =  repmat(struct('symbol',{}, 'size', {}, 'dtname', {}, 'baseaddr', {}), getNumParameters, 1 );
xcp.signals    =  repmat(struct('symbol',{}), getNumSignals, 1 );
xcp.models     =  cell(1,getNumModels);

    

   
   
         
xcp.events(1).id         = 0;
xcp.events(1).sampletime = 0.001;
xcp.events(1).offset     = 0.0;
         
xcp.events(2).id         = 1;
xcp.events(2).sampletime = 0.02;
xcp.events(2).offset     = 0.0;
         
xcp.events(3).id         = 2;
xcp.events(3).sampletime = 0.2;
xcp.events(3).offset     = 0.0;
         
xcp.events(4).id         = 3;
xcp.events(4).sampletime = 0.5;
xcp.events(4).offset     = 0.0;
         
xcp.events(5).id         = 4;
xcp.events(5).sampletime = 0.7;
xcp.events(5).offset     = 0.0;
         
xcp.events(6).id         = 5;
xcp.events(6).sampletime = 1.0;
xcp.events(6).offset     = 0.0;
         
xcp.events(7).id         = 6;
xcp.events(7).sampletime = 2.0;
xcp.events(7).offset     = 0.0;

function n = getNumParameters
n = 0;

function n = getNumSignals
n = 0;

function n = getNumEvents
n = 7;

function n = getNumModels
n = 1;

