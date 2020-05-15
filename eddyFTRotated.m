%
%   Date: 5/08/2020
%   
%   Description: 
%       Force-torque output script for eddy current manipulation of a
%       copper sphere with rotating dipole using CYLINDERICAL COORDINATE
%       system. 
%
%       Will output 6x3 matrix for force torque for the three possible
%       cylinderical positions 
%
%       Input:  
%           dists =  [m] distance from copper ball to dipole
%           dipole_strength = [Am^2] dipole strength of magnetic source
%           freq = [Hz] frequency of dipole rotation 
%           radii = [m] radii of copper sphere
%
%       Output:
%                        positions: rho, +z, -z
%           F_rho
%           F_z 
%           F_phi 
%           T_rho
%           T_z
%           T_phi
%% 

function [FT_Cyl, FT_Cartesian] = eddyFT( dist, dipole_strength, freq, radii)

% --- Inputs --- %
% dist = 0.100;           % [m]
% dipole_strength = 40;   % [Am^2]
% freq = 10;              % [Hz]
% radii = 0.05/2;         % [m]
% -------------- %

% Constants 
sig = 5.8e7;          % conductivity of copper 
mu = 4*pi*10^-7;        % Magnetic Permittivity of Free Space

% --- Model Coefficients --- % 
model_C = [...
    % --- Rho Configuration 
    % F Rho
    1 -1.27 -0.300 -7.26; ...               % Xaxis: Fx
    0 0 0  0;...                           % Xaxis: Fy = 0
    % F Phi 
    1 -1.13 -0.220 -7.22;...                % Xaxis: Fz
    0 0 0  0;...                           % Xaxis: Tx = 0
    % T X
    1 -1.44 -0.185 -6.23; ...               % Xaxis: Ty
    0 0 0  0;...                           % Xaxis: Tz
    % --- +Z Configuration
    % F Z
    1 -1.86 -0.238  -6.88 ;...              % Yaxis: Fy
    0 0 0  0; ...                          %  Yaxis: Fz
    0 0 0  0;...                           % Yaxis: Fx = 0

    % T Z 
    1 -2.05 -0.153  -5.67;...               % Yaxis: Ty
    0 0 0 0 ;...
    0 0 0  0;...                           % Yaxis: Tx = 0

     % --- -Z Configuration
    % F Z
    -1 -1.86 -0.238  -6.88 ;...              % Yaxis: Fy
    0 0 0  0; ...                          % Yaxis: Fz
    0 0 0 0 ;...                           % Yaxis: Fx = 0

    % T Z 
    1 -2.05 -0.153  -5.67;...               % Yaxis: Ty
    0 0 0 0;...
    0 0 0  0];                           % Yaxis: Tx = 0
 

% --- Calculate Force --- %
P1 = sig*mu*freq*radii^2;
P2 = dist/radii;
P0_temp = model_C(:, 1) .* ( 10.^( model_C(:,2).*abs(P1).^model_C(:,3) ) ) .*P2.^( model_C(:,4)); 
P0 = [P0_temp(1:6) , P0_temp(7:12) , P0_temp(13:18) ];
FT_Cyl = [P0(1:3,:).* mu.*dipole_strength.^2/(radii.^4); ...
    P0(4:6,:).* mu.*dipole_strength.^2/(radii.^3)]; 

FT_Cartesian = FT_Cyl; 
FT_Cartesian(3,1) = FT_Cartesian(3,1)*-1;
     
end 








