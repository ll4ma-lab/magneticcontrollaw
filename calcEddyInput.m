%
%   By: Lan Pham
%   Date: 9/2019
%
%
%   Calculates required input to achieve desired force-torque wrench on sphere based
%   on omnimagnet location, conductivty of materia (sigma) and outer
%   diameter of sphere. Also outputs the Actuation Matrix for the
%   omnimagnets
%
%   Originally developed to examine the conditioning (SVD) of different
%   omnimagnet configuration for controllablity of eddy current
%   manipulation
%
%

%
%   Inputs:
%       y_des           =   desired force torque wrench 
%       error_thresh    =   error threshold
%       omnimagnets     =   matrix of omnimagnet positions
%       sigma           =   sphere material conductivity 
%       do              =   sphere outer diameter 
%
%   Outputs:    
%       output_x        =   required input to get desired y output
%       output_error    =   error between desired y and actual y 
%       AllFT           =   actuation matrix 
%       AllFT_A         =   linear components of actuation matrix
%       AllFT_B         =   nonlinear components of actuation matrix
%       success         =   matrix of all successful inputs
%
%

function [output_x,output_error, AllFT, AllFT_A, AllFT_B, success] = ...
    calcEddyInput(y_des, error_thresh, omnimagnets, sigma, do)

mu = 4*pi*10^-7;            %Magnetic Permittivity of Free Space

% Matrix Pre-allocation
AllFT = zeros(6, 3*size(omnimagnets,2));
AllFT_A = AllFT; 
AllFT_B = AllFT; 

% Y-axis Modeling a (constant) and b (power) coefficients 
on_axis_F_abs = [4.03e-4 -7.23];
on_axis_T_lin = [6.36e-4 -6.35];
% X-axis Modeling 
off_axis_F_abs = [1.7e-3 -8];
off_axis_F_lin = [1.39e-3 -6.12];
off_axis_T_lin = [1.5e-3 -6.13];

% Build Matrices for Linear (A) components and Non-linear (B) components
coeff_A = [0 0 0 ; 0 0 -off_axis_F_lin(1); 0 off_axis_F_lin(1) 0; on_axis_T_lin(1) 0 0;...
    0 off_axis_T_lin(1) 0; 0 0 off_axis_T_lin(1)];
pow_A = [0 0 0 ; 0 0 off_axis_F_lin(2); 0 off_axis_F_lin(2) 0; on_axis_T_lin(2) 0 0;...
    0 off_axis_T_lin(2) 0; 0 0 off_axis_T_lin(2)];

coeff_B = [ on_axis_F_abs(1) off_axis_F_abs(1) off_axis_F_abs(1); 0 0 0 ; 0 0 0 ; 0 0 0 ; 0 0 0 ; 0 0 0 ];
pow_B =  [ on_axis_F_abs(2) off_axis_F_abs(2) off_axis_F_abs(2); 0 0 0 ; 0 0 0 ; 0 0 0 ; 0 0 0 ; 0 0 0 ];

omni_A = zeros(6,3);
omni_B = zeros(6,3); 

% Calculate the Modeled Input Matrix from Omnimagnet Locations 
for omni = 1:size(omnimagnets,1)
    pos = omnimagnets(omni,:)';
    dist = norm(pos);
    r = dist; 
    
    % Build master force-torque matrix dependent on location 
    for i = 1:6
        for j = 1:3
            if i < 4
                omni_A(i,j) = coeff_A(i,j)*(r/do)^pow_A(i,j)*mu^2*sigma/(do^2);
                omni_B(i,j)  = coeff_B(i,j)*(r/do)^pow_B(i,j)*mu^2*sigma/(do^2);
            else
                omni_A(i,j) = coeff_A(i,j)*(r/do)^pow_A(i,j)*mu^2*sigma/(do);
                omni_B(i,j) = coeff_B(i,j)*(r/do)^pow_B(i,j)*mu^2*sigma/(do);
            end
        end
    end
    
    omni_FT = omni_A + omni_B;
   
    % ----- Construct Rotation Vector from Omni Frame to World Frame ---%
    % X vector points toward tool
    omni_X = -1*pos/norm(pos);
    
    % Set default Z vector as upward
    Z = [0 0 1]';
    
    % Determine if default Z and X are the same
    if abs(omni_X) == abs(Z)
%         fprintf('X and Z are the same vectors, new perpendicular Z chosen [0 1 0]\n');
        Z = [0 1 0]';   %set Z to be something else if already equal
    end
    
    % Determine Z component perpendicular to Y axis
    if dot(omni_X,Z) == 0
%         fprintf('Z and X are already perpendicular\n');
        omni_Z = Z;
    else
        omni_Z = (omni_X./norm(omni_X))*sin( acos(dot(omni_X,Z)/(norm(omni_X)*norm(Z))));
    end
    
    % Obtain Y axis by crossing Z and X
    omni_Y = cross(omni_Z,omni_X);
    
    % Combine to build rotation matrix
    omniFrame = [omni_X, omni_Y, omni_Z];
    
    % Rotate Omnimagnet FT to the World Frame of the tool 
    omniworld_FT = [omniFrame*omni_FT(1:3,:); omniFrame*omni_FT(4:6,:)];
    omniworld_A  = [omniFrame*omni_A(1:3,:); omniFrame*omni_A(4:6,:)];
    omniworld_B  = [omniFrame*omni_B(1:3,:); omniFrame*omni_B(4:6,:)];
    
    % Keeps tracks the different rotations to each omnimagnet 
    omni_rots(:,:,omni) = omniFrame; 
    
    AllFT(:,(3*(omni-1) + 1):(3*omni)) = omniworld_FT;
    AllFT_A(:,(3*(omni-1) + 1):(3*omni)) = omniworld_A; 
    AllFT_B(:,(3*(omni-1) + 1):(3*omni)) = omniworld_B; 
    
end
%% 
% Solve Input (Required mt^2w) for Desired Force-Torque Wrench
%

% Iterative Process of Assuming +/- frequency for nonlinear A

numOmni = size(omnimagnets,1);
omniCombos = omniTable(numOmni); 
success = []; 
errors = [];
for test = 1:size(omniCombos,1)
    
    % Get Testing Set
    set = omniCombos(test,:); 
    set( set < 1) = -1;
    
    % Get Test Input
    temp_B = AllFT_B.*set;              % Get different input combination
    temp_all = AllFT_A + temp_B;        % Combine A and B Matrix for Solving
    temp_inv = pinv(temp_all);          % Solve temp actuation matrix
    temp_x_req = temp_inv*y_des;        % Solve for required input
    temp_lin = AllFT*temp_x_req;        % Check solution if system was linear (debug)
    temp_act = AllFT_A*temp_x_req + AllFT_B*abs(temp_x_req);    % Get actual wrench for solved X
    
    error = norm(temp_act-y_des)/norm(y_des);   % Get error
    
    if error <= error_thresh                         % Save solution if error is small
        
        success = [success, temp_x_req];
        errors = [errors, error];
        
        % Break after the first solution for large number of omnimagnets
        %         break;
    end 
end 

% Inform user if solution found 
if size(success,1) == 0
    fprintf('No Successful Solution Found\n')
    output_x = ones(1,size(AllFT,2))'*NaN;
else 
    fprintf('SOLUTION FOUND!\n')
    fprintf(strcat(num2str(size(success,2)),' Solution(s) Found','\n'))
    min_x = success(:,(errors == min(errors)));  % get solution with smallest 
    output_x = min_x(:,1);
    min_err = errors(errors == min(errors));
    output_error = min_err(1); 
end 



end % Function End



