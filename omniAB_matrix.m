%   Lan Pham
%   12/2019
%
%   Function that takes on the location of an omnimagnet, sphere
%   conductivity, sphere diameter and frequency of dipole rotation to outpu
%   the A and B matrix used to solve for eddy current  force and torque.
%   The omniFrame is the rotation
%
%   Inputs: 
%       omni_pos    =   Omnimagnet position relative to sphere (m)
%       sigma       =   conductivity of sphere
%       do          =   outer diameter of sphere (m)
%       freq        =   frequency of dipole rotations (Hz)
%
%   Outputs     
%       omni_A      =   The linear components of eddy current FT
%       omni_B      =   The non-linear or absolute direction components of
%                       FT
%       omniFrame   =   rotation matrix of the master omnimagnet
%                       configuration to the omni_pos location
%

function [omni_A, omni_B, omniFrame] = omniAB_matrix(omni_pos, sigma, Do, freq)

% check if pos is entered as omni_pos = [x, y, x] 
if size(omni_pos,1) == 3
    omni_pos = omni_pos'; 
end 

mu = 4*pi*10^-7;            %Magnetic Permittivity of Free Space

% Eddy Current Ansys Modeling 
model_C = [ ...
            % Off Axis Config (Xaxis)
            1 -2.86 -0.236 -0.00137 -7.39 ;...
            0 0 0 0 0;...
            1 -2.452 -0.129 -0.009395 -7.30007;...
            0 0 0 0 0;...
            1 -2.725 -0.118 -0.0111 -6.434; ...
            0 0 0 0 0;...
            % On Axis Config (Yaxis)
            0 0 0 0 0;...
            1 -3.639 -0.2126 0.002 -6.552 ;...
            0 0 0 0 0; ...
            0 0 0 0 0;...
            1 -3.353 -0.09175 -0.0012 -5.5076;...
            0 0 0 0 0];
       
% Organize for omnimagnet transformation 
% Y-axis Modeling a (constant) and b (power) coefficients
Yaxis_Fabs = model_C(6+2,2:end);
Yaxis_T = model_C(6+5,2:end);

% X-axis Modeling
Xaxis_Fabs = model_C(1,2:end);
Xaxis_Flin = model_C(3,2:end);
Xaxis_T = model_C(5,2:end);

% Build Matrices for Linear (A) components and Non-linear (B) components

% Linear:
C_A = zeros(6,3,5);
C_A(:,:,5) = zeros(6,3);
%X axis                                         Y axis                                          Z axis
0;                                              0;                                              0 ;                         % Fx
0;                                              0;                                              C_A(2,3,5) = 1; C_A(2,3,1:4) = Xaxis_Flin;          % Fy
0;                                              C_A(3,2,5)= -1 ; C_A(3,2,1:4) = Xaxis_Flin;     0;                          % Fz
C_A(4,1,5) = 1; C_A(4,1,1:4) = Yaxis_T;         0;                                              0;                                   % Tx
0;                                              C_A(5,2,5)= -1 ; C_A(5,2,1:4) = Xaxis_T;        0;                          % Ty
0;                                              0;                                              C_A(6,3,5) = -1; C_A(6,3,1:4) = Xaxis_T;           % Tz

% Absolute Terms
C_B = zeros(6,3,5);
C_B(:,:,5) = zeros(6,3);
C_B(1,1,1:4) = Yaxis_Fabs; C_B(1,2,1:4) = Xaxis_Fabs; C_B(1,3,1:4) = Xaxis_Fabs;
C_B(1,1,5) = 1; C_B(1,2,5) = 1; C_B(1,3,5) = 1;

omni_A = zeros(6,3);
omni_B = zeros(6,3);

% Calculate the Modeled Input Matrix from Omnimagnet Locations
pos = omni_pos';
dist = norm(pos);
D = dist;

P1 = sigma*mu*freq*Do^2;
P2 = D/Do;

% Build master force-torque matrix dependent on location
for i = 1:6
    for j = 1:3
        if i < 4
            % Force
            omni_A(i,j) = C_A(i,j,5)* ( 10^( C_A(i,j,1)*P1^C_A(i,j,2) ) * P2^( C_A(i,j,3)*P1+C_A(i,j,4)) ) * mu/(Do^4);
            omni_B(i,j) = C_B(i,j,5)* ( 10^( C_B(i,j,1)*P1^C_B(i,j,2) ) * P2^( C_B(i,j,3)*P1+C_B(i,j,4)) ) * mu/(Do^4);
        else
            %Torque
            omni_A(i,j) = C_A(i,j,5)* ( 10^( C_A(i,j,1)*P1^C_A(i,j,2) )*P2^(C_A(i,j,3)*P1+C_A(i,j,4)) ) * mu/(Do^3);
            omni_B(i,j) = C_B(i,j,5)* ( 10^( C_B(i,j,1)*P1^C_B(i,j,2) )*P2^(C_B(i,j,3)*P1+C_B(i,j,4)) ) * mu/(Do^3);
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

end %function end 