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
function [AllFT, AllFT_A, AllFT_B] = calcEddyInput(omnimagnets, sigma, DO, freq)
    % Matrix Pre-allocation
    AllFT_A = zeros(6, 3*size(omnimagnets,1)); 
    AllFT_B = AllFT_A; 


    % Calculate the Modeled Input Matrix from Omnimagnet Locations 
    for omni = 1:size(omnimagnets,1)
        pos = omnimagnets(omni,:)'
        [ omni_A, omni_B, omniFrame ]= omniAB_matrix(pos,sigma,DO,freq);
        
        % Rotate Omnimagnet FT to the World Frame of the tool 
        omniworld_A  = [omniFrame*omni_A(1:3,:); omniFrame*omni_A(4:6,:)];
        omniworld_B  = [omniFrame*omni_B(1:3,:); omniFrame*omni_B(4:6,:)];

        % Keeps tracks the different rotations to each omnimagnet 

        AllFT(:,(3*(omni-1) + 1):(3*omni)) = 5;
        AllFT_A(:,(3*(omni-1) + 1):(3*omni)) = omniworld_A; 
        AllFT_B(:,(3*(omni-1) + 1):(3*omni)) = omniworld_B; 

    end
end


