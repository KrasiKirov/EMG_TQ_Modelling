function upload(TG, sig, BUFdepth)
% UPLOAD - Uploads an array into Demo.mdl and then triggers display.
%    UPLOAD(TG, SIG, M) uploads array signal SIG using a buffer
%    array of M rows to the xPC Target application (TG), then outputs
%    the result.
%
%    Ver. 3.0

% Definitions -----------------------------------------------------------
BufferSig   = getparamid(TG, 'Buffer', 'Value');  % xPC Target parameter (Buffer)
CmdSig      = getparamid(TG, 'Command 1', 'Value'); % xPC Target parameter (CMD)
TrigSig     = getparamid(TG, 'Trigger 1', 'Value'); % xPC Target parameter (Trigger)

MAXBUFSIZE   = Inf;                     % Maximum buffer length allowable.
                                        % (Experimentally determined.)
% -----------------------------------------------------------------------

BUFwidth   = size(sig, 2);                    % No. columns in buffer array
SigDepth   = size(sig, 1);                    % No. rows in signal

ULStatusSig   = getsignalid(TG, 'Stim1/p2');     % xPC Target signal
ROStatusSig   = getsignalid(TG, 'Stim1/p3');    % xPC Target signal

NoAction      = ['setparam(TG, ' num2str(CmdSig)  ', 0);'];
LoadBuffer    = ['setparam(TG, ' num2str(CmdSig)  ', 1);'];
RstIndex      = ['setparam(TG, ' num2str(CmdSig)  ', 2);'];
GetReady      = ['setparam(TG, ' num2str(CmdSig)  ', 3);'];
GetAccumDepth = ['setparam(TG, ' num2str(CmdSig)  ', 4);'];
SetSigDepth   = ['setparam(TG, ' num2str(CmdSig)  ', 6);'];
GetSigDepth   = ['setparam(TG, ' num2str(CmdSig)  ', 7);'];
RstTrigger    = ['setparam(TG, ' num2str(TrigSig) ', 0);'];
TriggerOp     = ['setparam(TG, ' num2str(TrigSig) ', 1);'];
WriteBuffer   = ['setparam(TG, ' num2str(BufferSig) ', buffer);'];
ReadStatus    = ['getsignal(TG, ' num2str(ULStatusSig) ')'];

% Various status codes
BUSY        = 0;
DONE        = 1;
OP_FULL     = 2;
READY       = 3;
IGNORED     = 4;
ERROR       = 99;

% Some error checking
%
% 1) Check that xPC Target has successfully allocated memory for the
%    accumulator array.

rc = eval(ReadStatus);
if( rc == ERROR )
    error('xPC Target reports failure in allocating memory for accumulator');
end

% 2) Check buffer size
BufferSize = BUFdepth * BUFwidth;
if BufferSize > MAXBUFSIZE
    BufferSize
    error(['Buffer size (N*M) is limited to ' num2str(MAXBUFSIZE) '.'])
end

% 3) Check signal depth against accumulator size.
% Get the actual accumulator depth. (Remember that its negative will be returned.)
eval(GetReady)
while( eval(ReadStatus) ~= READY )
  pause(0.002)
end

eval(GetAccumDepth)                    % Query for information.
status = eval(ReadStatus);
while( status > 0 )
  %pause(0.001)
  status = eval(ReadStatus);
end
AccumDepth = -status;                  % Voila. No. rows in accumulator.
if( SigDepth > AccumDepth )
  error(' Signal length greater than accumulator array.')
end

% Inform the block of the actual signal depth.
disp('Setting the actual signal depth.')
eval(GetReady)
buffer  = zeros(BUFdepth, BUFwidth);
buffer(1,1) = SigDepth;
eval(WriteBuffer)
while( eval(ReadStatus) ~= READY )
  pause(0.001)
end
eval(SetSigDepth)
while( eval(ReadStatus) ~= DONE )
  pause(0.001)
end

% Reset things in case system is previously triggered.
%
disp('Disable trigger')                % Disable the trigger
eval(RstTrigger)                       %

disp('Resetting Sfun_BuildArray')
eval(RstIndex)                         % Get ready for upload
rc = eval(ReadStatus);
while( rc ~= DONE )
    %pause(0.001)
    rc = eval(ReadStatus);
end

disp('Uploading channel')

istart = 0;
while( istart < SigDepth )
    % Upload packets of stimulation signal(s) via the buffer.

    %disp('Get ready...')
    eval(GetReady)                     % Get ready for upload

    %disp('Load ...')
    k = min( [BUFdepth, SigDepth-istart] );
    buffer(1:k, :) = sig(istart+1:istart+k, :);
    
    % Wait for ready status.
    rc = eval(ReadStatus);
    while( rc ~= READY )
        %pause(0.001)
        rc = eval(ReadStatus);
    end   

    eval(WriteBuffer)
    eval(LoadBuffer)                   % Load buffer. Wait for completion.
    rc = eval(ReadStatus);
    while( rc == BUSY )
        %pause(0.001)
        rc = eval(ReadStatus);
    end

    istart = istart + BUFdepth;
end % while

disp('Triggering output display')
eval(TriggerOp)

% Do some reporting
disp(' ')
disp(['Accumulator depth: ' num2str(AccumDepth)])

eval(GetSigDepth)                      % Query for information.
status = eval(ReadStatus);
while( status > 0 )
  %pause(0.001)
  status = eval(ReadStatus);
end
SigDepth = -status;                  % Voila. No. rows in accumulator.
disp(['Signal depth: ' num2str(SigDepth)])