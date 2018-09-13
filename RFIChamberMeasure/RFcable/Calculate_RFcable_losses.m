clear()
close all
%-------------------------------------------------------------------------------
%                                  LOAD DATA (S-PARAMETERS)
%-------------------------------------------------------------------------------
Filename = 'C:/Users/geomarr/Documents/GitHub/set-up-GUI/RFIChamberMeasure/RFcable/sparametersRFcable/S21circuitLossesReal_Imag.s2p';

n = 1;
FreqStart = 100;
FreqEnd = 7000;
N_Pts = 201;
% Load the S21 data (Insertion loss) and S11 (Return loss)
[S11,S21,S12,S22,F] = ZVBSegFRead(Filename,FreqStart,FreqEnd,N_Pts);
F = F/1e9;
figure();
plot(F,(abs(S11)));
ylabel('VSWR');
xlabel('Frequency (GHz)');
title('Return Loss');
grid();

figure();
plot(F, 10*log10(abs(S21)));
title('S_{21}');
ylabel('Magnitude (dB)');
xlabel('Frequency (GHz)');
grid();
gain = [F, 10*log10(abs(S21))];
csvwrite('S21circuitLossesReal_Imag.csv', gain, delimited = '')
#Filename = 'C:/Users/geomarr/Desktop/RFIChamberMeasure/RFcable/sparametersRFcable/S21circuitLossesReal_Imag.s2p';

% Load the S21 data (Insertion loss) and S11 (Return loss)
#[S11,S21,S12,S22,F] = ZVBSegFRead(Filename,FreqStart,FreqEnd,N_Pts);
#F = F/1e9;

#figure();
#plot(F, 10*log10(abs(S21)));
#title('Insertion Loss (Attenuation)');
#ylabel('Magnitude (dB)');
#xlabel('Frequency (GHz)');
#grid();