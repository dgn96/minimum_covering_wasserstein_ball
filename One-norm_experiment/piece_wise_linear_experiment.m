close all;
x = linspace(-1, 1, 2^10);
n = 3;
shifts = (1-2*rand(n,1))*1;
lambda = 1*rand(n,1);
x_hat = zeros(2^10, 1)';
for i=1:n
    plot(x, lambda(i)*abs(x - shifts(i)))
    hold on

    x_hat = x_hat + lambda(i)*abs(x - shifts(i));
end

figure
plot(x, x_hat)


