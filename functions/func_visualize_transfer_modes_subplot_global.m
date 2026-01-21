function func_visualize_transfer_modes_subplot_global(V_all, state_centers_all, legend_labels, selected_indices)
    % Create a figure window
    figure('Name', 'Transfer Modes Visualization', 'Position', [100, 100, 1800, 700]);
    
    % First, find the global min and max values across all data
    global_min = Inf;
    global_max = -Inf;
    
    % Loop through selected systems to find global min/max
    for j = 1:length(selected_indices)
        i = selected_indices(j);
        V_sorted = V_all{i};
        
        % Extract all eigenvectors that will be plotted
        rho1 = V_sorted(:, 1);
        realrho2 = real(V_sorted(:, 2));
        imagrho2 = imag(V_sorted(:, 2));
        realrho3 = real(V_sorted(:, 3));
        
        % Combine all modes for this system
        all_data = [rho1; realrho2; imagrho2; realrho3];
        
        % Update global min and max
        global_min = min(global_min, min(all_data));
        global_max = max(global_max, max(all_data));
    end
    
    % Create a custom colormap for all panels
    zero_pos = (0 - global_min) / (global_max - global_min);
    zero_pos = max(0, min(1, zero_pos));
    n_colors = 256;
    n_negative = round(zero_pos * n_colors);
    n_positive = n_colors - n_negative;
    negative_color = [0, 0, 1];  % Blue for negative values
    positive_color = [1, 0, 0];  % Red for positive values
    
    if n_negative > 0
        neg_red = linspace(negative_color(1), 1, n_negative);
        neg_green = linspace(negative_color(2), 1, n_negative);
        neg_blue = linspace(negative_color(3), 1, n_negative);
        negative_part = [neg_red', neg_green', neg_blue'];
    else
        negative_part = [];
    end
    
    if n_positive > 0
        pos_red = linspace(1, positive_color(1), n_positive);
        pos_green = linspace(1, positive_color(2), n_positive);
        pos_blue = linspace(1, positive_color(3), n_positive);
        positive_part = [pos_red', pos_green', pos_blue'];
    else
        positive_part = [];
    end
    
    custom_cmap = [negative_part; positive_part];
    colormap(custom_cmap);
    
    % Now plot all panels
    t_idx = 1;
    for j = 1:length(selected_indices)
        i = selected_indices(j);
        V_sorted = V_all{i};
        
        % Eigenvectors
        rho1 = V_sorted(:, 1);
        realrho2 = real(V_sorted(:, 2));
        imagrho2 = imag(V_sorted(:, 2));
        realrho3 = real(V_sorted(:, 3));
        
        % Titles and modes
        mode_list = {rho1, realrho2, imagrho2, realrho3};
        label_list = {'Eigenvector 1', ...
                     'Real part of eigenvector 2', ...
                     'Imag part of eigenvector 2', ...
                     'Real part of eigenvector 3'};
        
        for k = 1:4
            subplot(2, 4, t_idx);
            func_visualize_transfer_modes_panel_global(mode_list{k}, state_centers_all{i}, legend_labels{i}, label_list{k}, global_min, global_max);
            t_idx = t_idx + 1;
        end
    end
    
    % Add a single colorbar for the entire figure
    h = colorbar('Position', [0.93 0.1 0.02 0.8]);
    ylabel(h, 'Value', 'FontSize', 10);
    caxis([global_min, global_max]);
end

function func_visualize_transfer_modes_panel_global(V, state_centers_ulam_active, system_name, mode_label, global_min, global_max)
    % Plot using global color limits
    max_abs = max(abs(V));
    alpha_data = (abs(V) / max_abs).^(1/2);
    scatter3(state_centers_ulam_active(:,1), ...
             state_centers_ulam_active(:,2), ...
             state_centers_ulam_active(:,3), ...
             20, V, 'filled', ...
             'MarkerFaceAlpha', 'flat', ...
             'AlphaData', alpha_data);
    
    axis tight; view(3); box on;
    title(sprintf('%s\n%s', system_name, mode_label), 'FontSize', 10);
    set(gca, 'FontSize', 8);
    
    % Set consistent color limits for all panels
    caxis([global_min, global_max]);
end
