clear all; close all; clc;
addpath('./equations/'); % Ensure all model functions are in this folder
addpath('./functions/')

%% Global Settings
options = odeset('AbsTol',1e-7,'RelTol',1e-7);
Time = 99999.9;
% Time = 250000.0;
plot_transfer_modes = false;

%% Define model filenames (without `.m`)
model_files = {
    'sprott13original', ...
    'sprott13example1', 'sprott13example2', ...
    'sprott13example3', 'sprott13example4', ...
    'sprott13example5', 'sprott13example6', ...
};


%% Prepare legend labels
legend_labels = {
    'Sprott13', ...
    'example1', 'example2', ...
    'example3', 'example4', ...
    'example5', 'example6', ...
};

%% Storage for eigenvalues
eigvals_all = {}; D_all = {}; V_all = {}; eigvecs_all = {}; state_centers_all = {};
system_states = {};

%% Loop over all models
for i = 1:length(model_files)
    model_name = model_files{i};
    fprintf("Processing: %s...\n", model_name);

    % Get function handle
    fh = str2func(model_name);

    % Compute transfer operator
    [rho, ~, ~, ~, state_centers, V, D, y] = func_compute_transfer_operator_1(fh, Time, options);
    D_all{i} = D;
    V_all{i} = V;
    state_centers_all{i} = state_centers;
    system_states{i} = y;

    % Sort eigenvalues and eigenvectors together
    [eigvals_sorted, idx] = sort(abs(diag(D)), 'descend');
    eigvals_all{i} = eigvals_sorted;
    V_all{i} = V(:, idx);  % Sort eigenvectors accordingly
end

%%
% Choose indices of the systems you want to plot
subset_indices = [1, 2]; % Lorenz + S_r=0.2 #1-2 + S_r=0.3 #1-4

plot_length = 300;
figure;
hold on;
colors = lines(length(subset_indices));

for j = 1:length(subset_indices)
    i = subset_indices(j);
    if i == 1
        % Make original Lorenz thicker and black
        plot(1:plot_length, eigvals_all{i}(1:plot_length), '-', ...
            'LineWidth', 2.5, 'Color', [0 0 0]);
    else
        plot(1:plot_length, eigvals_all{i}(1:plot_length), '-', 'LineWidth', 1.5, 'Color', colors(j,:));
    end
end
xlabel('Eigenvalue Index', 'FontSize', 12);
ylabel('|Eigenvalue|', 'FontSize', 12);
title('Subset of Systems: Lorenz + Selected S_r', 'FontSize', 14);
legend(legend_labels(subset_indices), 'Location', 'northeast');
grid on;

%% Save data for future subset plots
% save('./save_data/sprott13_main.mat', 'eigvals_all', 'D_all', 'V_all', 'state_centers_all');


%% 
% % Choose indices of the systems you want to plot
subset_indices = [1, 4]; % Lorenz + S_r=0.2 #1-2 + S_r=0.3 #1-4

plot_length = 600;
figure;
hold on;
colors = lines(length(subset_indices));

for j = 1:length(subset_indices)
    i = subset_indices(j);
    if i == 1
        % Make original Lorenz thicker and black
        plot(1:plot_length, eigvals_all{i}(1:plot_length), '-', ...
            'LineWidth', 2.5, 'Color', [0 0 0]);
    else
        plot(1:plot_length, eigvals_all{i}(1:plot_length), '-', 'LineWidth', 1.5, 'Color', colors(j,:));
    end
end
xlabel('Eigenvalue Index', 'FontSize', 12);
ylabel('|Eigenvalue|', 'FontSize', 12);
title('Subset of Systems: Lorenz + Selected S_r', 'FontSize', 14);
legend(legend_labels(subset_indices), 'Location', 'northeast');
grid on;

%%
selected_indices = [1, 5];  % Your selected systems
func_visualize_transfer_modes_subplot(V_all, state_centers_all, legend_labels, selected_indices);


% use a gloabl color map
selected_indices = [1, 2];  % Your selected systems
func_visualize_transfer_modes_subplot_global(V_all, state_centers_all, legend_labels, selected_indices);
