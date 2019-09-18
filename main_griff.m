%
%   Example script for using the calcEddyInput Function 
%
%
%   To Griffin,
%       I think you will only need the AllFT_A and AllFT_B matrix for your
%       solver 
%

% ------- CONSTANTS -------- %
sigma_copper = 5.8E+07;
sigma_al = 3.77E+07;
sigma = sigma_copper;
do = 0.05;
% -------------------------- %

% --- Omnimagnet Configurations --- %
% Fully Surrounded Configuration 
% omnimagnets = 0.001 * [-130 0 0 ; 0 0 -130; 130 0 0; 0 0 130; 0 130 0 ; 0 -130 0]; 
% Open Side Configuration
omnimagnets = 0.001 * [-130 0 0 ; 0 0 -130; 130 0 0; 0 130 0 ; 0 -130 0];


y_des = [1 0 0 0 0 0]';                                 % desired force-torque wrench
error_thresh = 0.05;                                    % error threshold (currently 5%)
wrench_independence = zeros(size(omnimagnets,1)*3, 6);  % keeps track if there are solutions 
check_wrench = zeros(6, 6);                             % checks of output gives accurate desired y
errors = zeros(1, 6); 

% Iterate through each isolating wrench
for a = 1:size(y_des,1)
    % Calculate required Eddy Current Input
    [output_x, error, AllFT, AllFT_A, AllFT_B, success] = calcEddyInput(y_des, error_thresh, omnimagnets, sigma, do);
    wrench_independence(:,a) = output_x; 
    check_wrench(:,a) = AllFT_A*output_x + AllFT_B*abs(output_x); 
    errors(:,a) = error; 
    y_des = circshift(y_des,1);                         % shift the desired force/wrench to next one
end 



