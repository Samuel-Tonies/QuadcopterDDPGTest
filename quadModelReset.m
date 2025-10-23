function [InitialObs,InitialState] = quadModelReset()
% Reset function to place initial states


% Current place at origin for testing


% x0 = rand() * 0.1;
% y0 = rand() * 0.1;
% z0 =rand() * 0.1;
% xd0 = rand() * 0.01;
% yd0 = rand() * 0.01;
% zd0 = rand() * 0.01;
% phi0 = rand() * 0.01;
% theta0 = rand() * 0.01;
% psi0 = rand() * 0.01;
% phid0 = rand() * 0.001;
% thetad0 = rand() * 0.001;
% psid0 = rand() * 0.001;

% InitialState = [x0; xd0; y0; yd0; z0; zd0; phi0; phid0; theta0; thetad0; psi0; psid0];
% InitialObs = InitialState;

InitialState = zeros(12,1);
InitialObs = InitialState;
end

