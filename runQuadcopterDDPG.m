%% Runs Training for quadcopter

% Sets rng for debugging
rng("default")

% Define Actor Info
% Action bounded from 0-1 then scaled to RPM from 0-500
actInfo = rlNumericSpec([1 1]);
actInfo.Name = 'action_signal';
actInfo.LowerLimit = 0;
actInfo.UpperLimit = 1;

% Define Obs Info
% x, x_dot, y, ... phi, phi_dot, theta,...
obsInfo = rlNumericSpec([12 1]);
obsInfo.Name = 'plant_states';
obsInfo.LowerLimit = -10^3;
obsInfo.UpperLimit = 10^3;


% Initialize state
State = zeros(12,1);

% Create Environment
env = rlFunctionEnv(obsInfo,actInfo,"quadModelStep","quadModelReset");

% Time step
Ts = 0.01;
% Final time of each episode
Tf = 15;

% Actor network
actorNetwork = [
    featureInputLayer(12, 'Name', 'observation')
    fullyConnectedLayer(32, 'Name', 'actor_fc1')
    reluLayer('Name', 'actor_relu1')
    fullyConnectedLayer(32, 'Name', 'actor_fc2')
    reluLayer('Name', 'actor_relu2')
    fullyConnectedLayer(1, 'Name', 'actor_output')   % single scalar action
    sigmoidLayer('Name', 'actor_sigmoid')            % keep 0..1 output
];

% --- Simple critic (obs path 32, action path 16, then concat -> 32) ---
% observation path
criticObs = [
    featureInputLayer(12, 'Name', 'observation')
    fullyConnectedLayer(32, 'Name', 'critic_obs_fc1')
    reluLayer('Name', 'critic_obs_relu1')
];

% action path (small)
criticAct = [
    featureInputLayer(1, 'Name', 'action')
    fullyConnectedLayer(16, 'Name', 'critic_action_fc1')
    reluLayer('Name', 'critic_action_relu1')
];

% after concatenation
criticPost = [
    concatenationLayer(1, 2, 'Name', 'critic_concat')   % concat on first dim
    fullyConnectedLayer(32, 'Name', 'critic_fc1')
    reluLayer('Name', 'critic_relu1')
    fullyConnectedLayer(1, 'Name', 'critic_output')
];

% assemble layer graph
criticLG = layerGraph(criticObs);
criticLG = addLayers(criticLG, criticAct);
criticLG = addLayers(criticLG, criticPost);

% connect action path into concatenation input 2
criticLG = connectLayers(criticLG, 'critic_action_relu1', 'critic_concat/in2');

% --- Representation / optimizer options (kept simple) ---
actorOptions = rlOptimizerOptions('LearnRate',1e-5);
criticOptions = rlOptimizerOptions('LearnRate',1e-4);

% actor = rlDeterministicActorRepresentation(actorNetwork, obsInfo, actInfo, ...
%     'Observation', {'observation'}, 'Action', {'actor_sigmoid'}, actorOptions);
% 
% critic = rlQValueRepresentation(criticLG, obsInfo, actInfo, ...
%     'Observation', {'observation'}, 'Action', {'action'}, criticOptions);

actor = rlDeterministicActorRepresentation(actorNetwork, obsInfo, actInfo, ...
    'Observation', {'observation'}, 'Action', {'actor_sigmoid'}, actorOptions);

critic = rlQValueRepresentation(criticLayerGraph, obsInfo, actInfo, ...
    'Observation', {'observation'}, 'Action', {'action'}, criticOptions);

opt = rlDDPGAgentOptions;
opt.NoiseOptions = rl.option.GaussianActionNoise;
opt.NoiseOptions.StandardDeviation = 0.22;
opt.NoiseOptions.StandardDeviationDecayRate = 1e-5;
opt.NoiseOptions.StandardDeviationMin = 0.1;
opt.NoiseOptions.Mean = 0;

% Create agent
agent = rlDDPGAgent(actor, critic,opt);

% Agent options
agent.AgentOptions.SampleTime = Ts;
agent.AgentOptions.ResetExperienceBufferBeforeTraining = true;
agent.UseExplorationPolicy = true;

% Exploration Parameters
% agent.AgentOptions.NoiseOptions.StandardDeviation = 0.25;
% agent.AgentOptions.NoiseOptions.StandardDeviationDecayRate = 1e-5;
% agent.AgentOptions.NoiseOptions.StandardDeviationMin = 0.01; 
% possibly too small for exploration but want to avoid pushing to max/min
% action
% agent.AgentOptions.NoiseOptions.Mean = -0.5;
% agent.AgentOptions.NoiseOptions.InitialAction = 0.45;
% agent.AgentOptions.NumWarmStartSteps = 200;
% agent.AgentOptions.LearningFrequency = 200;

maxsteps = ceil(Tf/Ts); 
trainOpts = rlTrainingOptions(...
    MaxEpisodes= 30000,...
    MaxStepsPerEpisode= maxsteps,...
    Plots="training-progress",...
    StopTrainingCriteria="AverageReward",...
    StopTrainingValue= -1,... % set large for testing
    SaveAgentCriteria="EpisodeReward",...
    SaveAgentValue= -50);
% trainOpts.UseParallel = true;
% trainOpts.ParallelizationOptions.Mode = "async";

trainingStats = train(agent,env,trainOpts);

% Run simulation of agent
simOptions = rlSimulationOptions(MaxSteps=maxsteps);
experience = sim(env,agent,simOptions);

% Plot experience
quadPlot(experience);