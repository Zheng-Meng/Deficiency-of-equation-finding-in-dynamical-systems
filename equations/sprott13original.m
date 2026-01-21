function u = sprott13original(t, x)
    u = [
        -2 * x(2);
        x(1) + x(3)^2;
        1 + x(2) - 2 * x(3)
    ];
end
