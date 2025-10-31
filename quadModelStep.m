function [NextObs, Reward, IsDone,NextState] = quadModelStep(Action, State)
% Define physical parameters
g=9.81;
b=3.25e-5;
d=7.5e-7;
l=0.25;
m=0.65;
ix=7.5e-3;
iy=7.5e-3;
iz=1.3e-2;
Ts = 0.01; % note that Ts should be same as Ts in runQuadcopterDDPG

% Unpack State
x = State(1);
xd = State(2);
y = State(3);
yd = State(4);
z = State(5);
zd = State(6);
phi = State(7);
phid = State(8);
theta = State(9);
thetad = State(10);
psi = State(11);
psid = State(12);

% Define action scaling
max_w = 500;
w_scale = max_w;

% Unpack Action
w1 = Action(1)*w_scale;
w2 = Action(2)*w_scale;
w3 = Action(3)*w_scale;
w4 = Action(4)*w_scale;

% w2 = Action(2)*w_scale;
% w3 = Action(3)*w_scale;
% w4 = Action(4)*w_scale;

% Ensure input bounded
w1 = max(min(w1,max_w),0);
w2 = max(min(w2,max_w),0);
w3 = max(min(w3,max_w),0);
w4 = max(min(w4,max_w),0);

% disp([w1 w2 w3 w4])
% Calculate thrusts and torques
u1=b*((w1^2)+(w2^2)+(w3^2)+(w4^2));
u2=b*l*((w3^2)-(w1^2));
u3=b*l*((w4^2)-(w2^2));
u4=d*((w2^2)+(w4^2)-(w1^2)-(w3^2));

% EoMs
xdd=(u1/m)*(sin(phi)*sin(psi)+cos(phi)*cos(psi)*sin(theta));
ydd=(u1/m)*(cos(phi)*sin(psi)*sin(theta)-cos(psi)*sin(phi));
zdd=((u1/m)*(cos(phi)*cos(theta)))-g;

phidd  =(((iy-iz)/ix)*thetad*psid)+(u2/ix);
thetadd=(((iz-ix)/iy)*phid*psid)+(u3/iy);
psidd  =(((ix-iy)/iz)*phid*thetad)+(u4/iz);

% Integration
new_xd=(xdd*Ts)+xd;
new_yd=(ydd*Ts)+yd;
new_zd=(zdd*Ts)+zd;

new_x =(new_xd*Ts)+x;
new_y =(new_yd*Ts)+y;
new_z =(new_zd*Ts)+z;

new_phid=(phidd*Ts)+phid;
new_thetad=(thetadd*Ts)+thetad;
new_psid=(psidd*Ts)+psid;

new_phi =(new_phid*Ts)+phi;
new_theta =(new_thetad*Ts)+theta;
new_psi =(new_psid*Ts)+psi;

State_dd = [xdd,ydd,zdd,phidd,thetadd,psidd];
% Concatenate states
NextState = [new_x; new_xd; new_y; new_yd; new_z; new_zd; new_phi; new_phid; new_theta; new_thetad; new_psi; new_psid];

% Agent observation is next state
NextObs = NextState;

% Reward calculation
[Reward, IsDone] = rewardCalc(NextState, State, State_dd, Action);

end