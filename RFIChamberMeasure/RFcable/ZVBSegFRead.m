%-------------------------------------------------------------------------------
%                   ZVB CW Mode Data Reading
%                   ------------------------------
% Filename  : *.s2p file from CW Mode measurement on ZVB-14
%
function [S11,S21,S12,S22,F] = ZVBSegFRead(Filename,FreqStart,FreqEnd,N_Pts);
  
  Increm = int8((FreqEnd-FreqStart)/N_Pts);
  S11 = zeros(1:FreqEnd-FreqStart,Increm);
  S21 = zeros(1:FreqEnd-FreqStart,Increm);
  S12 = zeros(1:FreqEnd-FreqStart,Increm);
  S22 = zeros(1:FreqEnd-FreqStart,Increm);


  
  DataReg = importdata(Filename, ' ', 5);
  % Load frequency
  F = DataReg.data(1:N_Pts,1);
    % Load the N_Pts S21, S11 and S22 data for one frequency
  S11 = DataReg.data(1:N_Pts,2) + 1i*DataReg.data(1:N_Pts,3);
  S21 = DataReg.data(1:N_Pts,4) + 1i*DataReg.data(1:N_Pts,5);
  S12 = DataReg.data(1:N_Pts,6) + 1i*DataReg.data(1:N_Pts,7);
  S22 = DataReg.data(1:N_Pts,8) + 1i*DataReg.data(1:N_Pts,9);
    
end
  