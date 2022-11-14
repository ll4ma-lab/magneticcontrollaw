import numpy as np
import math


def ForceFinder(w,m,material,FEA_Exp,r,lambda1):



    mu0 = 4*math.pi*10**-7
    m_loc_start = np.array([[lambda1, 0, 0]])
    obj_loc_start = np.array([0, 0, 0])


    # Determine Object's mass and sigma
    if material == 'aluminum':
        sigma = 3.69e7;
    elif material == 'copper':
        sigma = 5.87e7;
    elif material == 'titanium':
        sigma = 2.4e6;


    # Define Model Constants       
    if FEA_Exp == 1:
        c0p90 = 266
        c1p90 = 2.6
        c2p90 = -0.101
        c3p90 = -7.65
        c4p90 = 7
        c0p0 = 460
        c1p0 = 2.95
        c2p0 = -0.101
        c3p0 = -9.26
        c4p0 = 7
        c0phi = 6040
        c1phi = 3.45
        c2phi = -0.102
        c3phi = -14.3
        c4phi = 7
        c0tp = 6840
        c1tp = 3.0
        c2tp = -0.0986
        c3tp = -13.2
        c4tp = 6
        c0ttheta = 8100
        c1ttheta = 3.6
        c2ttheta = -0.0985
        c3ttheta = -15.7
        c4ttheta = 6
    elif FEA_Exp == 0:
        c0p90 = 282
        c1p90 = 3.2
        c2p90 = -0.098
        c3p90 = -9.41
        c4p90 = 7
        c0p0 = 467
        c1p0 = 2.81
        c2p0 = -0.0969
        c3p0 = -9.75
        c4p0 = 7
        c0phi = 5870
        c1phi = 3.49
        c2phi = -0.0973
        c3phi = -14.6
        c4phi = 7
        c0tp = 6900
        c1tp = 3.35
        c2tp = -0.099
        c3tp = 14.9
        c4tp = 6
        c0ttheta = 8000
        c1ttheta = 3.4
        c2ttheta = -0.0928
        c3ttheta = -15
        c4ttheta = 6
     
    what = w/np.linalg.norm(w)
    p = obj_loc_start - m_loc_start

    # Calculate the Theta value for each magnet relative to the object
    if np.linalg.norm(w) == w[0]:
        theta = 0
    elif np.linalg.norm(w) == -w[0]:
        theta = math.pi
    else:           
        theta = math.atan2(np.linalg.norm(np.cross(what,p)),np.dot(what,np.transpose(p)))
        
    # Construct unit rho, theta, and phi vectors
    ip = p/np.linalg.norm(p);
    if np.linalg.norm(np.cross(what,p)) != 0:
        iphi = np.cross(what,p)/np.linalg.norm(np.cross(what,p));
    else:
        iphi = float("nan")
    if np.isnan(iphi).any():
        itheta = float("nan")
    else:
        itheta = np.cross(iphi,ip);
        
    # Calculate the non-dimensional terms
    Pi1 = sigma*mu0*np.linalg.norm(w)*r**2
    Pi2 = np.linalg.norm(p)/r
    Pi0_p90 = ((c0p90*Pi1)**(c1p90*Pi1**c2p90)*10**c3p90)/Pi2**c4p90
    Pi0_p0 = ((c0p0*Pi1)**(c1p0*Pi1**c2p0)*10**c3p0)/Pi2**c4p0
    Pi0_phi900 = ((c0phi*Pi1)**(c1phi*Pi1**c2phi)*10**c3phi)/Pi2**c4phi
    Pi0_tp = ((c0tp*Pi1)**(c1tp*Pi1**c2tp)*10**c3tp)/Pi2**c4tp;
    Pi0_ttheta = ((c0ttheta*Pi1)**(c1ttheta*Pi1**c2ttheta)*10**c3ttheta)/Pi2**c4ttheta;
    
    
    
    # Calculate the p=0 p=90 rho and phi forces
    f_p90 = Pi0_p90*mu0/r**4
    f_p0 = Pi0_p0*mu0/r**4
    f_phi90 = Pi0_phi900*mu0/r**4
    t_p0 = Pi0_tp*mu0/r**3;
    t_theta90 = Pi0_ttheta*mu0/r**3;
    
    
    # Calculate the Forces at theta location
    f_p = -(f_p90 - f_p0)/2*math.cos(2*theta) + (f_p90 + f_p0)/2
    f_phi = f_phi90*math.sin(theta)
    t_p = t_p0*math.cos(theta);
    t_theta = t_theta90*math.sin(theta);
    
    # Calculate the total force from each omnimagnet
    if np.isnan(iphi).any():
        f = (f_p*np.transpose(ip))*m**2;
    else:
        f = (f_p*np.transpose(ip) + f_phi*np.transpose(iphi))*m**2;
    if np.isnan(itheta).any():
        t = (t_p*np.transpose(ip))*m**2;
    else:
        t = (t_p*np.transpose(ip) + t_theta*np.transpose(itheta))*m**2;


    
    return f, abs(f_phi)/abs(f_p), t
        
 