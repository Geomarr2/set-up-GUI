%-------------------------------------------------------------------------------
Filename_1GHz = 'C:\Users\geomarr\Desktop\RFIChamberMeasure\SNR\Inject different RF freq\test-Signal.mat';
Filename_1GHz_noise = 'C:\Users\geomarr\Desktop\RFIChamberMeasure\SNR\Inject different RF freq\3GHz_noise.mat';
Filename_1GHz_csv = 'C:\Users\geomarr\Desktop\RFIChamberMeasure\SNR\Inject different RF freq\test-Signal.csv';

DataReg = load(Filename_1GHz);
data = csvread(Filename_1GHz_csv, 128,0);
Z0 = 119.9169832*pi;                                                            # Impedance of freespace
Signal_at_1GHz = 20*log10(1000*((abs(DataReg.Y).^2)/Z0));
true_BW = 56e+006;                                                              # Sampling frequency
samples = length(data);
CFreq = DataReg.InputCenter;
BW = DataReg.Span;                                                              # Acquire Bandwith
RBW = round(BW/samples);                                                        # resolution bandwith
StartFreq = CFreq - BW/2;
StopFreq = CFreq + BW/2;

freq = StartFreq:BW/samples:StopFreq-1;

plot(freq', data);