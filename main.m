% MAIN SCRIPT TO ANALYZE AND VISUALIZE TRANSFER OPERATORS FOR LORENZ & SINDY

clear all;close all; clc;
addpath('./equations/'); % Ensure all functions are on path
addpath('./functions/')

%% Global Settings
options = odeset('AbsTol',1e-7,'RelTol',1e-7);
Time = 99999.9;
% Time = 9999.9;
plot_transfer_modes = false;

%% 1. True Lorenz System
fprintf("Processing: True Lorenz system...\n");
[rho0, realrho0b, imagrho0b, realrho0c, state_centers0, V0, D0] = func_compute_transfer_operator(@(t,y) lorenz63original(t,y), Time, options);
if plot_transfer_modes == true
    func_visualize_transfer_modes(rho0, realrho0b, imagrho0b, realrho0c, state_centers0, 'Lorenz', 0);
end

%% 2. SINDy Model 1
fprintf("Processing: SINDy Model 1...\n");
[rho1, realrho1b, imagrho1b, realrho1c, state_centers1, V1, D1] = func_compute_transfer_operator(@lorenz63sindy1, Time, options);
if plot_transfer_modes == true
    func_visualize_transfer_modes(rho1, realrho1b, imagrho1b, realrho1c, state_centers1, 'SINDy 1', 10);
end
%% 3. SINDy Model 2
fprintf("Processing: SINDy Model 2...\n");
[rho2, realrho2b, imagrho2b, realrho2c, state_centers2, V2, D2] = func_compute_transfer_operator(@lorenz63sindy2, Time, options);
if plot_transfer_modes == true
    func_visualize_transfer_modes(rho2, realrho2b, imagrho2b, realrho2c, state_centers2, 'SINDy 2', 20);
end
%% 4. SINDy Model 3
fprintf("Processing: SINDy Model 3...\n");
[rho3, realrho3b, imagrho3b, realrho3c, state_centers3, V3, D3] = func_compute_transfer_operator(@lorenz63sindy3, Time, options);
if plot_transfer_modes == true
    func_visualize_transfer_modes(rho3, realrho3b, imagrho3b, realrho3c, state_centers3, 'SINDy 3', 30);
end

%% Eigenvalue analysis
eigvals0 = diag(D0);  % Lorenz
[eigvals_sort0, ~] = sort(abs(eigvals0), 'descend');

eigvals1 = diag(D1);  % SINDy 1
[eigvals_sort1, ~] = sort(abs(eigvals1), 'descend');

eigvals2 = diag(D2);  % SINDy 2
[eigvals_sort2, ~] = sort(abs(eigvals2), 'descend');

eigvals3 = diag(D3);  % SINDy 3
[eigvals_sort3, ~] = sort(abs(eigvals3), 'descend');

plot_length_eig = 600;
markersize = 1;
linewidth = 1;
% Define custom colors (RGB)
color0 = [0.0, 0.45, 0.74]; % Blue (Lorenz)
color1 = [0.85, 0.33, 0.10]; % Red-Orange (SINDy 1)
color2 = [0.47, 0.67, 0.19]; % Green (SINDy 2)
color3 = [0.49, 0.18, 0.56]; % Purple (SINDy 3)

% Plot
figure;
hold on
plot(1:plot_length_eig, eigvals_sort0(1:plot_length_eig), 'o-', 'LineWidth', linewidth, 'MarkerSize', markersize, 'Color', color0);
plot(1:plot_length_eig, eigvals_sort1(1:plot_length_eig), 's-', 'LineWidth', linewidth, 'MarkerSize', markersize, 'Color', color1);
plot(1:plot_length_eig, eigvals_sort2(1:plot_length_eig), '^-', 'LineWidth', linewidth, 'MarkerSize', markersize, 'Color', color2);
plot(1:plot_length_eig, eigvals_sort3(1:plot_length_eig), '*-', 'LineWidth', linewidth, 'MarkerSize', markersize, 'Color', color3);
xlabel('Eigenvalue Index', 'FontSize', 12);
ylabel('|Eigenvalue|', 'FontSize', 12);
title('Sorted Magnitudes of Eigenvalues', 'FontSize', 14);
legend({'Lorenz', 'SINDy 1', 'SINDy 2', 'SINDy 3'}, 'Location', 'northeast');
grid on;

% Plot
figure;
hold on
plot(1:plot_length_eig, eigvals_sort0(1:plot_length_eig), 'o-', 'LineWidth', linewidth, 'MarkerSize', markersize, 'Color', color0);
plot(1:plot_length_eig, eigvals_sort1(1:plot_length_eig), 's-', 'LineWidth', linewidth, 'MarkerSize', markersize, 'Color', color1);
xlabel('Eigenvalue Index', 'FontSize', 12);
ylabel('|Eigenvalue|', 'FontSize', 12);
title('Sorted Magnitudes of Eigenvalues', 'FontSize', 14);
legend({'Lorenz', 'SINDy 1'}, 'Location', 'northeast');
grid on;

% Plot
figure;
hold on
plot(1:plot_length_eig, eigvals_sort0(1:plot_length_eig), 'o-', 'LineWidth', linewidth, 'MarkerSize', markersize, 'Color', color0);
plot(1:plot_length_eig, eigvals_sort2(1:plot_length_eig), '^-', 'LineWidth', linewidth, 'MarkerSize', markersize, 'Color', color2);
xlabel('Eigenvalue Index', 'FontSize', 12);
ylabel('|Eigenvalue|', 'FontSize', 12);
title('Sorted Magnitudes of Eigenvalues', 'FontSize', 14);
legend({'Lorenz', 'SINDy 2'}, 'Location', 'northeast');
grid on;

% Plot
figure;
hold on
plot(1:plot_length_eig, eigvals_sort0(1:plot_length_eig), 'o-', 'LineWidth', linewidth, 'MarkerSize', markersize, 'Color', color0);
plot(1:plot_length_eig, eigvals_sort3(1:plot_length_eig), '*-', 'LineWidth', linewidth, 'MarkerSize', markersize, 'Color', color3);
xlabel('Eigenvalue Index', 'FontSize', 12);
ylabel('|Eigenvalue|', 'FontSize', 12);
title('Sorted Magnitudes of Eigenvalues', 'FontSize', 14);
legend({'Lorenz',  'SINDy 3'}, 'Location', 'northeast');
grid on;

% Plot 1,2
figure;
hold on
plot(1:plot_length_eig, eigvals_sort1(1:plot_length_eig), 's-', 'LineWidth', linewidth, 'MarkerSize', markersize, 'Color', color1);
plot(1:plot_length_eig, eigvals_sort2(1:plot_length_eig), '^-', 'LineWidth', linewidth, 'MarkerSize', markersize, 'Color', color2);

xlabel('Eigenvalue Index', 'FontSize', 12);
ylabel('|Eigenvalue|', 'FontSize', 12);
title('Sorted Magnitudes of Eigenvalues', 'FontSize', 14);
legend({'SINDy 1', 'SINDy 2'}, 'Location', 'northeast');
grid on;

% Plot 2,3
figure;
hold on
plot(1:plot_length_eig, eigvals_sort2(1:plot_length_eig), '^-', 'LineWidth', linewidth, 'MarkerSize', markersize, 'Color', color2);
plot(1:plot_length_eig, eigvals_sort3(1:plot_length_eig), '*-', 'LineWidth', linewidth, 'MarkerSize', markersize, 'Color', color3);
xlabel('Eigenvalue Index', 'FontSize', 12);
ylabel('|Eigenvalue|', 'FontSize', 12);
title('Sorted Magnitudes of Eigenvalues', 'FontSize', 14);
legend({'SINDy 2', 'SINDy 3'}, 'Location', 'northeast');
grid on;

%%
% save eigenvalues 
% save('eigenvalues_main_lorenz.mat', 'eigvals_sort0', 'eigvals_sort1', 'eigvals_sort2', 'eigvals_sort3');

