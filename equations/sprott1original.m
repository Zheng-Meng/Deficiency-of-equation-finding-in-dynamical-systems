function u = sprott1original(t, x)
    u = [
        x(2) * x(3);
        x(1) - x(2);
        1 - x(1) * x(2)
    ];
end
