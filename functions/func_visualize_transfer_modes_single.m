function func_visualize_transfer_modes_single(V, state_centers_ulam_active, system_name, mode_label)
    % Plot a scalar field V on a 3D grid with color & transparency

    figure()
    data_min = min(V(:));
    data_max = max(V(:));

    % Determine color position of zero
    zero_pos = (0 - data_min) / (data_max - data_min);
    zero_pos = max(0, min(1, zero_pos));

    n_colors = 256;
    negative_color = [0, 0, 1];
    positive_color = [1, 0, 0];

    n_negative = round(zero_pos * n_colors);
    n_positive = n_colors - n_negative;

    % Create blue-to-white and white-to-red segments
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

    max_abs = max(abs(V));
    alpha_data = (abs(V) / max_abs).^(1/2);

    scatter_h = scatter3(state_centers_ulam_active(:,1), state_centers_ulam_active(:,2), state_centers_ulam_active(:,3), 200, V, 'filled');
    colormap(custom_cmap);
    caxis([data_min, data_max]);
    scatter_h.AlphaData = alpha_data;
    scatter_h.MarkerFaceAlpha = 'flat';
    set(gcf, 'Renderer', 'opengl');

    xlabel('X'); ylabel('Y'); zlabel('Z');
    title(sprintf('%s - %s', system_name, mode_label));
    colorbar;
end