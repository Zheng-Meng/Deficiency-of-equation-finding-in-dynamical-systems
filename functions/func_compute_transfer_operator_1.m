function [rho0, realrho0b, imagrho0b, realrho0c, state_centers_ulam_active, V, D, y] = ...
    func_compute_transfer_operator_1(odefun, Time, options)
    % with larger dt
    % Ulam grid definition
    grid_step = 0.05;
    % [X, Y, Z] = ndgrid(-20:grid_step:20, -28:grid_step:28, 0:grid_step:50);
    [X, Y, Z] = ndgrid(0:grid_step:1, 0:grid_step:1, 0:grid_step:1);
    state_centers_ulam = [X(:), Y(:), Z(:)];

    % Warm-up integration
    [~, y_temp] = ode45(odefun, 0:0.375:1000, randn(3,1)' * 0.01, options);
    ci = y_temp(end,:) + randn(1,3) * 0.01;

    % Long integration
    [~, y] = ode45(odefun, 0:0.375:Time, ci, options);
    % y_save = y;
    % % every 10 points, take one point
    % y = y(1:10:end,:);
    f = 100;
    siz = size(y,1);

    % Normalize trajectory to [0, 1] range per dimension
    y_min = min(y(:,1:3), [], 1);
    y_max = max(y(:,1:3), [], 1);
    y_norm = (y(:,1:3) - y_min) ./ (y_max - y_min + 1e-10);  % Add small number to avoid division by zero

    % Assign trajectory points to Ulam boxes
    state_labels = [];
    for j = 1:f
        start_idx = round((j-1)*siz/f + 1);
        end_idx   = round(j*siz/f);
        idx = start_idx:end_idx;
        [~, labels] = pdist2(state_centers_ulam, y_norm(idx,:), 'euclidean', 'Smallest', 1);
        state_labels = [state_labels labels];
    end

    % Transition matrix construction
    num_states = size(state_centers_ulam,1);
    T = sparse(num_states, num_states);
    for t = 1:length(state_labels)-1
        T(state_labels(t), state_labels(t+1)) = T(state_labels(t), state_labels(t+1)) + 1;
    end

    % Remove unused states
    rows_nonzero = sum(T,2) > 0;
    cols_nonzero = sum(T,1) > 0;
    T_red = T(rows_nonzero, cols_nonzero);
    state_centers_ulam_active = state_centers_ulam(rows_nonzero,:);

    % Row-normalize
    PF = full(T_red ./ sum(T_red, 2));
    PF = PF';

    % Eigendecomposition
    [V, D] = eig(PF);
    eigvals = diag(D);
    [~, idx_sorted] = sort(abs(eigvals), 'descend');

    rho0 = V(:, idx_sorted(1));
    realrho0b = real(V(:, idx_sorted(2)));
    imagrho0b = imag(V(:, idx_sorted(2)));
    realrho0c = real(V(:, idx_sorted(3)));
end