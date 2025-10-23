function quadPlot(experience)
%Extract observations/action/time steps
plant_states = experience.Observation.plant_states.Data;
action_signal = experience.Action.action_signal.Data;
t = experience.Observation.plant_states.Time;

% Plots states from last experience
x = squeeze(plant_states(1,1,:));
xd = squeeze(plant_states(2,1,:));
y = squeeze(plant_states(3,1,:));
yd = squeeze(plant_states(4,1,:));
z = squeeze(plant_states(5,1,:));
zd = squeeze(plant_states(6,1,:));
phi = squeeze(plant_states(7,1,:));
phid = squeeze(plant_states(8,1,:));
theta = squeeze(plant_states(9,1,:));
thetad = squeeze(plant_states(10,1,:));
psi = squeeze(plant_states(11,1,:));
psid = squeeze(plant_states(12,1,:));

w1 = squeeze(action_signal(1,1,:));
w2 = squeeze(action_signal(2,1,:));
w3 = squeeze(action_signal(3,1,:));
w4 = squeeze(action_signal(4,1,:));

figure
plot(t, x)
hold on
plot(t, y)
plot(t, z)
grid on
title("Position vs Time")
legend("x","y","z")
ylabel("Position [m]")
xlabel("Time [s]")

figure
plot(t, phi)
hold on
plot(t, theta)
plot(t, psi)
grid on
title("Attitude vs Time")
legend("phi","theta","psi")
ylabel("Attitude [rad]")
xlabel("Time [s]")

figure
plot(t, xd)
hold on
plot(t, yd)
plot(t, zd)
grid on
title("Velocity vs Time")
legend("xd","yd","zd")
ylabel("Velocity [m/s]")
xlabel("Time [s]")

figure
plot(t, phid)
hold on
title("Angular Velocity vs Time")
plot(t, thetad)
plot(t, psid)
grid on
ylabel("Angular Velocity [rad/s]")
xlabel("Time [s]")
legend("phid","thetad","psid")

% Plots zero control input on first time step
figure
plot(t(1:end-1), w1)
hold on
plot(t(1:end-1), w2)
plot(t(1:end-1), w3)
plot(t(1:end-1), w4)
grid on
title("Action vs Time")
legend("w1","w2","w3","w4")
ylabel("Rotor Action [RPM]")
xlabel("Time [s]")

figure
experience.Reward.plot
end

