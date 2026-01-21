clear all; close all; clc;
addpath('./equations/'); % Ensure all model functions are in this folder
addpath('./functions/')

%% Global Settings
options = odeset('AbsTol',1e-7,'RelTol',1e-7);
Time = 9999.9;
% Time = 500000.0;
plot_transfer_modes = false;

%% Define model filenames (without `.m`)
model_files = {
    'lorenz63original', ...
    'lorenz63example1', ...
    'lorenz63example2', 'lorenz63example3', ...
    'lorenz63example4', 'lorenz63example5', 'lorenz63example6', 'lorenz63example7', ...
    'lorenz63example8', 'lorenz63example9',...
};

%% Prepare legend labels
legend_labels = {
    'Lorenz', ...
    'example1', ...
    'example2', 'example3', 'example4', ...
    'example5', 'example6', 'example7', 'example8', 'example9'
};

%% Storage for eigenvalues
eigvals_all = {}; D_all = {}; V_all = {}; eigvecs_all = {}; state_centers_all = {};

%% Loop over all models
for i = 1:length(model_files)
    model_name = model_files{i};
    fprintf("Processing: %s...\n", model_name);

    % Get function handle
    fh = str2func(model_name);

    % Compute transfer operator
    [rho, ~, ~, ~, state_centers, V, D] = func_compute_transfer_operator(fh, Time, options);
    D_all{i} = D;
    V_all{i} = V;
    state_centers_all{i} = state_centers;

    % Sort eigenvalues and eigenvectors together
    [eigvals_sorted, idx] = sort(abs(diag(D)), 'descend');
    eigvals_all{i} = eigvals_sorted;
    V_all{i} = V(:, idx);  % Sort eigenvectors accordingly
end

%% Plot eigenvalue magnitudes - Full Figure
plot_length = 600;
markersize = 1;
linewidth = 1;
colors = lines(length(model_files));

figure;
hold on;
for i = 1:length(eigvals_all)
    if i == 1
        % Make original Lorenz thicker and black
        plot(1:plot_length, eigvals_all{i}(1:plot_length), '-', ...
            'LineWidth', 2.5, 'Color', [0 0 0]);
    else
        plot(1:plot_length, eigvals_all{i}(1:plot_length), '-', ...
            'LineWidth', linewidth, 'MarkerSize', markersize, 'Color', colors(i,:));
    end
end
xlabel('Eigenvalue Index', 'FontSize', 12);
ylabel('|Eigenvalue|', 'FontSize', 12);
title('Sorted Magnitudes of Eigenvalues for All Systems', 'FontSize', 14);
legend(legend_labels, 'Location', 'northeastoutside');
grid on;


% Choose indices of the systems you want to plot
subset_indices = [1, 3, 4, 5, 6, 7, 8]; % Lorenz + S_r=0.2 #1-2 + S_r=0.3 #1-4

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
% Choose indices of the systems you want to plot
subset_indices = [1, 5]; 

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

%% Save data for future subset plots
% save('eigenvalues_lorenz_sindy.mat', 'eigvals_all', 'legend_labels', 'model_files');


%% 
% Choose indices of the systems you want to plot
subset_indices = [6, 8]; % Lorenz + S_r=0.2 #1-2 + S_r=0.3 #1-4

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

%% Plot transfer modes for selected indices
selected_indices = [1, 4];  % Example: Lorenz + two SINDy models
for j = 1:length(selected_indices)
    i = selected_indices(j);
    fprintf("Visualizing modes for: %s\n", legend_labels{i});

    V_sorted = V_all{i};
    rho1 = V_sorted(:, 1);
    rho2 = V_sorted(:, 2);
    rho3 = V_sorted(:, 3);

    realrho1 = real(V_sorted(:, 1));
    imagrho1 = imag(V_sorted(:, 1));
    realrho2 = real(V_sorted(:, 2));
    imagrho2 = imag(V_sorted(:, 2));
    realrho3 = real(V_sorted(:, 3));
    imagrho3 = imag(V_sorted(:, 3));

    func_visualize_transfer_modes_single(rho1, state_centers_all{i}, legend_labels{i}, 'eigenvector 1');
    func_visualize_transfer_modes_single(realrho2, state_centers_all{i}, legend_labels{i}, 'Real part of eigenvector 2');
    func_visualize_transfer_modes_single(imagrho2, state_centers_all{i}, legend_labels{i}, 'Imag part of eigenvector 2');
    func_visualize_transfer_modes_single(realrho3, state_centers_all{i}, legend_labels{i}, 'Real part of eigenvector 3');
end


%% 

selected_indices = [1, 7];  % Your selected systems
func_visualize_transfer_modes_subplot(V_all, state_centers_all, legend_labels, selected_indices);


% use a gloabl color map
% selected_indices = [1, 2];  % Your selected systems
func_visualize_transfer_modes_subplot_global(V_all, state_centers_all, legend_labels, selected_indices);



