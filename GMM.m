clear,clc
% 随机生成50个点
data = randn(150, 2);

% 初始化GMM参数
K = 3; % 假设有2个高斯分布
N = size(data, 1);
mu = randn(K, 2);
Sigma = repmat(eye(2), [1, 1, K]);
pi = ones(K, 1) / K;

% EM迭代
for iter = 1:100
    % E步骤
    gamma = zeros(N, K);

    for k = 1:K
        gamma(:, k) = pi(k) * mvnpdf(data, mu(k, :), Sigma(:, :, k));
    end

    gamma = gamma ./ sum(gamma, 2);

    % M步骤
    Nk = sum(gamma, 1);

    for k = 1:K
        mu(k, :) = gamma(:, k)' * data / Nk(k);
        Sigma(:, :, k) = (data - mu(k, :))' * diag(gamma(:, k)) * (data - mu(k, :)) / Nk(k);
        pi(k) = Nk(k) / N;
    end

end

% 输出结果
figure;
scatter3(data(:, 1), data(:, 2), zeros(size(data, 1), 1), 'filled', 'r');

% 为每个高斯分布画出概率密度函数
hold on;
[X, Y] = meshgrid(linspace(min(data(:, 1)), max(data(:, 1)), 100), linspace(min(data(:, 2)), max(data(:, 2)), 100));

for k = 1:K
    Z = mvnpdf([X(:) Y(:)], mu(k, :), Sigma(:, :, k));
    Z = reshape(Z, size(X));
    surf(X, Y, Z, 'EdgeColor', 'none', 'FaceAlpha', 0.6);
end

hold off;
