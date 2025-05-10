close all;
clear all;

basis = ["H", "V", "D", "A", "L", "R"];
pa_1__port = serialport('COM9', 115200);   % PA
pa_2__port = serialport('COM13', 115200);   % PA

% Waveshaper IP
ip = '10.0.3.16';
% Define profile according to API
profileType = 'bandpass';   % blockall, transmit, bandpass, bandstop, gaussian
centerFreq = 193.73;        % THz
bandwidth = 4.38;          % THz
attenuation = 0;            % dB
port = 1;

% bandwidth_range = [4.380, 2.190, 1.095, 0.548, 0.274, 0.137, 0.068, 0.034, 0.017, 0.009];
% 
% for f=bandwidth_range
%     bandwidth = f;
%     % Upload predefined profile using above definition
%     r = uploadPredefinedProfile(ip, profileType, centerFreq, bandwidth, attenuation, port);
% 
%     for d1=1:6
%         for d2=1:6
%             coinc_mat(d1, d2) = measure_basis(pa_1__port, pa_2__port, basis(d1), basis(d2));    
%         end
%     end
% 
%     coinc_vec = reshape(coinc_mat, 1, []);      % A row vector of 36 coincidence counts
%     display(coinc_vec); 
%     file = fopen('Alice_frequency_tracking_coincidence_counts.txt', 'a');
%     fprintf(file, f, coinc_vec);
% end

r = uploadPredefinedProfile(ip, profileType, centerFreq, bandwidth, attenuation, port);
coinc_mat = measure_basis(pa_1__port, pa_2__port, "H", "H");

%% PA Analysis
function Coinc = measure_basis(port_1, port_2, x, y)
writeline(port_1, "2");
writeline(port_1, "c");
writeline(port_1, x);

writeline(port_2, "2");
writeline(port_2, "c");
writeline(port_2, y);

tacq = 30; % integration time
Start_offset = -1000; % start histogram offset
end_offset = 1000; % end histogram offset
start_coinc_bin = 1126; % start coincidence sum offset
% Start_offset = 134500; % start histogram offset
% end_offset = 136500; % end histogram offset
% start_coinc_bin = 63; % start coincidence sum offset
window = 21;
end_coinc_bin = start_coinc_bin + (window-1); % end coincidence sum offset
%end_coinc_bin = start_coinc_bin + 20; % end coincidence sum offset
%%
sp = ' '; % no change
site1name = 'a'; % a: Alice
site1ip = '10.0.3.5'; % Alice IP address
site2name = 'b'; % b: Bob
site2ip = '10.0.3.6'; % Bob IP address 
countA_cmd = [pwd, '\main_count.exe']; % no change
countB_cmd = [pwd, '\main_count.exe']; % no change
StartCMD = ['cmd_two.bat ', countA_cmd, sp, site1ip, sp, num2str(tacq), sp, site1name, sp, countB_cmd, sp, site2ip, sp, num2str(tacq), sp, site2name]; % no change
%%
find_coinc = 1; % 0=Plot histograms; 1=Find coinc
%% Coincidence counting command
CoincCMD = ['main_coinc_mat.exe ', ... % no change
    num2str(Start_offset), ' ', ... % no change
    num2str(end_offset), ' ', ... % no change
    [site1name, '1'], ' ', ... % should be 1 for det 1
    [site2name, '1'], ' ', ... % should be 2 for det 2
    ];

[~, ~] = system(StartCMD);
pause(tacq+2);

[~, cmdout] = system(CoincCMD);
cmdout_data = strsplit(cmdout, ',');
%%
file_names = 'coinc_a1b1_'; % add the name of expected histograms. Change _a1a1_ to refelect the detectors
%%
% Define range locations
max_sum = 0;
max_file = '';
max_hist = [];
max_pos = 0;
m_pos = 0;
m_val = 0;
values = [];
if find_coinc == 1
    for i = 1:3
        data = csvread([file_names,num2str(i-2)]);
        values = data(3:end);
        [m_val,m_pos] = max(values);
        sum_values = sum(values(start_coinc_bin:end_coinc_bin));
        if sum_values > max_sum
            max_sum = sum_values;
            max_file = num2str(i-2);
            max_hist = values;
            max_pos = m_pos;
        end
    end
    % Display results
    fprintf('Max sum: %d\n', max_sum);
    fprintf('Associated file: %s\n', max_file);
    plot(max_hist,"lineWidth",3,'DisplayName',num2str(i-2))
    xlim([1 length(max_hist)])
    legend
else
    figure;
    hold on;
    for i = 1:3
        data = csvread([file_names,num2str(i-2)]);
        values = data(3:end);
        plot(values,"lineWidth",3,'DisplayName',['PPS ffset: ' num2str(i-2)])
        [m_val,m_pos] = max(values);
        sum_values = sum(values(start_coinc_bin:end_coinc_bin));
        if sum_values > max_sum
            max_sum = sum_values;
            max_file = num2str(i-2);
            max_hist = values;
            max_pos = m_pos;
        end
    end
    xlim([1 length(max_hist)])
    legend
end
%%
countA = str2double(cmdout_data{1});
countB = str2double(cmdout_data{2});
Acc = sum(values(1:21));
Coinc = max_sum;

%file = fopen('Alice_frequency_tracking_coincidence_counts.txt', 'a');
fprintf('Det A: %d\tDet B: %d\tAcc: %d\tCoinc: %d \tPeak: %d\n',countA, countB, Acc, Coinc, max_pos);
end

%% Waveshaper function
function r = uploadPredefinedProfile(ip, profileType, center, bandwidth, attn, port, timeout)
    % Create json payload
    data.type = profileType;
    data.port = port;
    data.center = center;
    data.bandwidth = bandwidth;
    data.attn = attn;
    jsonString = jsonencode(data);
    
    if(~exist('timeout','var'))
        timeout = 5;
    end
    
    % POST data and decode returned JSON
    options = weboptions('Timeout', timeout);
    r = webwrite(['http://', ip, '/waveshaper/loadprofile'], jsonString, options);
    r = jsondecode(r);
end