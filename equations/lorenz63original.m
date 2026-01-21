function u = lorenz63(t,x)
    s = 10;
    r = 28;
    b = 8/3;
    u = [s*(x(2)-x(1)), x(1)*(r - x(3)) - x(2), x(1)*x(2) - b*x(3)]';
end
