%% Runs Training for quadcopter

% Sets rng for debugging
rng("default")

% Define Actor Info
% Action bounded from 0-1 then scaled to RPM from 0-500
actInfo = rlNumericSpec([4 1]);
actInfo.Name = 'action_signal';
actInfo.LowerLimit = 0;
actInfo.UpperLimit = 1;

% Define Obs Info
% x, x_dot, y, ... phi, phi_dot, theta,...
obsInfo = rlNumericSpec([12 1]);
obsInfo.Name = 'plant_states';

% Initialize state
State = zeros(12,1);

% Create Environment
env = rlFunctionEnv(obsInfo,actInfo,"quadModelStep","quadModelReset");

% Time step
Ts = 0.1;
% Final time of each episode
Tf = 15;

% Actor network
actorNetwork = [
    featureInputLayer(12, 'Name', 'observation')
    fullyConnectedLayer(256, 'Name', 'actor_fc1')
    reluLayer('Name', 'actor_relu1')
    fullyConnectedLayer(256, 'Name', 'actor_fc2')
    reluLayer('Name', 'actor_relu2')
    fullyConnectedLayer(4, 'Name', 'actor_output')
    tanhLayer('Name', 'actor_tanh')  % Output between -1 and 1
    scalingLayer('Name', 'actor_scaling', 'Scale', 0.5, 'Bias', 0.5) % Scale to 0-1
];

% Critic network
criticNetwork = [
    featureInputLayer(12, 'Name', 'observation')
    fullyConnectedLayer(256, 'Name', 'critic_obs_fc1')
    reluLayer('Name', 'critic_obs_relu1')
    concatenationLayer(1, 2, 'Name', 'critic_concat')
    fullyConnectedLayer(256, 'Name', 'critic_fc1')
    reluLayer('Name', 'critic_relu1')
    fullyConnectedLayer(256, 'Name', 'critic_fc2')
    reluLayer('Name', 'critic_relu2')
    fullyConnectedLayer(1, 'Name', 'critic_output')
];

% Create layer graph for critic
criticLayerGraph = layerGraph(criticNetwork);

actionPath = [
    featureInputLayer(4, 'Name', 'action')
    fullyConnectedLayer(256, 'Name', 'action_fc1')
    reluLayer('Name', 'action_relu1')
];

criticLayerGraph = addLayers(criticLayerGraph, actionPath);
criticLayerGraph = connectLayers(criticLayerGraph, 'action_relu1', 'critic_concat/in2');

% Create representations
actorOptions = rlOptimizerOptions('LearnRate', 1e-4);
criticOptions = rlOptimizerOptions('LearnRate', 1e-3);

actor = rlDeterministicActorRepresentation(actorNetwork, obsInfo, actInfo, ...
    'Observation', {'observation'}, 'Action', {'actor_scaling'}, actorOptions);

critic = rlQValueRepresentation(criticLayerGraph, obsInfo, actInfo, ...
    'Observation', {'observation'}, 'Action', {'action'}, criticOptions);

% Create agent
agent = rlDDPGAgent(actor, critic);

% Agent options
agent.AgentOptions.SampleTime = Ts;
agent.AgentOptions.ResetExperienceBufferBeforeTraining = true;
agent.UseExplorationPolicy = true;

% Exploration Parameters
agent.AgentOptions.NoiseOptions.StandardDeviation = 0.1;
agent.AgentOptions.NoiseOptions.StandardDeviationDecayRate = 1e-5;
agent.AgentOptions.NoiseOptions.StandardDeviationMin = 0.005; 
% possibly too small for exploration but want to avoid pushing to max/min
% action
agent.AgentOptions.NoiseOptions.Mean = 0;
agent.AgentOptions.NoiseOptions.InitialAction = 0.45;
% agent.AgentOptions.NumWarmStartSteps = 200;
% agent.AgentOptions.LearningFrequency = 200;

maxsteps = ceil(Tf/Ts); 
trainOpts = rlTrainingOptions(...
    MaxEpisodes= 30000,...
    MaxStepsPerEpisode= maxsteps,...
    Plots="training-progress",...
    StopTrainingCriteria="AverageReward",...
    StopTrainingValue= 1000000,... % set large for testing
    SaveAgentCriteria="EpisodeReward",...
    SaveAgentValue=10000);
% trainOpts.UseParallel = true;
% trainOpts.ParallelizationOptions.Mode = "async";

trainingStats = train(agent,env,trainOpts);

% Run simulation of agent
simOptions = rlSimulationOptions(MaxSteps=500);
experience = sim(env,agent,simOptions);

% Plot experience
quadPlot(experience);