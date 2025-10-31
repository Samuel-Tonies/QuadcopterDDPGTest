function [Reward, IsDone] = rewardCalc(NextState,State, State_dd, Action)
% Calculate Reward function

% Unpack State(t-1)
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

% Unpack State(t)
next_x = NextState(1);
next_xd = NextState(2);
next_y = NextState(3);
next_yd = NextState(4);
next_z = NextState(5);
next_zd = NextState(6);
next_phi = NextState(7);
next_phid = NextState(8);
next_theta = NextState(9);
next_thetad = NextState(10);
next_psi = NextState(11);
next_psid = NextState(12);

zdd = State_dd(3); % Attempting to reduce large changes in thrust

% Episodes are terminated if excessively large states are reached
IsDone = any([abs(next_z) abs(next_x) abs(next_y)] > 10)...
    || any([abs(next_phi) abs(next_theta) abs(next_psi)] > pi*2);

 % Currently set as simple reward function  for testing
    % Reward = -0.00001*(next_phid^2 + next_thetad^2 + next_psid^2) - 0.1*(next_zd^2) + 500;
    Reward = -(next_zd^2) + -0.1*(zdd^2);
    % Small positive reward is given at each time step to avoid getting
    % stuck in local maxima where agent tries to crash as soon as possible
    % to terminate episode early and avoid accumulating more negative
    % rewards
   
% if IsDone
%     Reward = Reward - 10; 
%     % When IsDone is true, a large negative reward is given for "crashing"
% end

% Check for Inf/NaN rewards due to massive state values
if ~isfinite(Reward)
    Reward = -1*10^50;
    IsDone = true; % Terminate episode if reward is not finite
end

end

