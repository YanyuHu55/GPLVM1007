import torch
import numpy as np
import math
import torch.optim as optim
from torch.distributions.multivariate_normal import MultivariateNormal
import math
from sklearn.cluster import KMeans
from torch.distributions import kl_divergence
import argparse
import matplotlib.pyplot as plt

torch.autograd.set_detect_anomaly(True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('current device: ', device)

# parameter in terminal
params = argparse.ArgumentParser()
params.add_argument('-num_iterations', type=int, default=10, help='iteration time ')
params.add_argument('-learning_rate', type=float, default=0.002, help='learning rate')
params.add_argument('-output_scale_generate', type=float, default=1., help='output scale for generating Y')
params.add_argument('-length_scale_generate', type=float, default=1., help='length scale for generating Y')
params.add_argument('-noise_generate', type=float, default=0.01, help='noise variance for generating Y')
params.add_argument('-output_scale_model', type=float, default=2., help='initialized output scale for model')
params.add_argument('-length_scale_model', type=float, default=2., help='initialized length scale for model')
params.add_argument('-noise_model', type=float, default=1., help='initialized noise variance for model')
params.add_argument('-data_points', type=int, default=5, help='number of data points')
args = params.parse_args()

num_iterations = args.num_iterations
learning_rate = args.learning_rate
output_scale_generate = args.output_scale_generate
length_scale_generate = args.length_scale_generate
noise_generate = args.noise_generate
output_scale_model = args.output_scale_model
length_scale_model = args.length_scale_model
noise_model = args.noise_model
data_points = args.data_points

# ARD kernel
# def squared_exponential_kernel(X1, X2, length_scale, output_scale):
#     pairwise_distances = torch.cdist(X1 / length_scale, X2 / length_scale, p=2).pow(2)
#     kernel_values = output_scale ** 2 * torch.exp(-0.5 * pairwise_distances)
#     return kernel_values

# RBF kernel
def squared_exponential_kernel(x1, x2, length_scale, output_scale):
    pairwise_sq_dists = torch.sum((x1[:, None] - x2) ** 2, dim=-1)

    return output_scale ** 2 * torch.exp(-pairwise_sq_dists / (2 * (length_scale ** 2)))


# def initialize_inducing_inputs(X, M, random_seed=42):
#     # print('X', X)
#     torch.manual_seed(random_seed)
#     indices = torch.randperm(X.size(0))[:M]
#     inducing_inputs = X[indices]
#     # print('inducing points', inducing_inputs)
#     return inducing_inputs

def initialize_inducing_inputs(X, M, random_seed=42):
    # print('X', X)
    torch.manual_seed(random_seed)

    # inducing point = kmeans method
    kmeans = KMeans(n_clusters=M, random_state=random_seed)
    # print('kmeans',kmeans)
    kmeans.fit(X.cpu())
    # print('kmeans.fit(X)', kmeans.fit(X))
    inducing_inputs = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)
    return inducing_inputs

def train_Y(X, output_scale_layer1, length_scale_layer1, noise_variance):

    # training input function value
    # P(F(X)) = N( m(X), K1(X,X)=K_nn ) mean and variance
    # m(X) = 0
    mean_vector_X = torch.zeros(len(X), 1).to(device)
    K_nn = squared_exponential_kernel(
        X, X, length_scale_layer1, output_scale_layer1
    )
    print('Knn of Y',K_nn)

    p_Y = torch.distributions.MultivariateNormal(mean_vector_X.reshape(1, -1).squeeze(0),
                                                 K_nn + noise_variance ** 2 * torch.eye(len(X)).to(device))
    Y = p_Y.sample((1,)).reshape(-1, 1)
    return Y

def KL_divergence(mean_p_x, mean_q_x, var_p_x, var_q_x):
    '''
    KL divergence KL(q(X)||p(x))
    KL(q(x) || p(x)) = 0.5 * [ ln(det(Σ_p) / det(Σ_q)) + tr(Σ_p⁻¹ * Σ_q) + (μ_p - μ_q)ᵀ * Σ_p⁻¹ * (μ_p - μ_q) - k]

    '''

    q_x = torch.distributions.Normal(mean_q_x, torch.sqrt(var_q_x))
    p_x = torch.distributions.Normal(mean_p_x, torch.sqrt(var_p_x))
    kl_per_latent_dim = kl_divergence(q_x, p_x).sum(axis=0)
    KL_q_p = kl_per_latent_dim.sum()

    return KL_q_p

def fi0_function(N, output_scale_layer1):
    '''
    fi(0) is a scaler,
    fi(0) = sum_n^N(fi_n(0)) = N * outputscale**2
    '''
    return N * (output_scale_layer1 ** 2)

def fi1_function(N, M, Q, output_scale_layer1, w_q, mean_q_x, var_q_x, Z):
    '''
    fi_1 is a N*M matrix
    fi_1_nm =
    '''
    # print('Z',Z)
    # print('mean_q_x',mean_q_x)
    fi_1 = torch.zeros(N, M).double().to(device)
    fraction_multiply_term = 1
    for q in range(Q):
        miu_nq_minus_zmq = torch.cdist(mean_q_x[:,q].reshape(-1, 1), Z[:, q].reshape(-1, 1), p=2).pow(2)
        # print('miu_nq_minus_zmq',miu_nq_minus_zmq)
        # print('miu_nq_minus_zmq', miu_nq_minus_zmq.shape)
        exp_term = - 0.5 * (w_q * miu_nq_minus_zmq) / (w_q * var_q_x[:, q].reshape(-1, 1) + 1)
        # print('exp_term',exp_term)
        denominator = torch.sqrt(w_q * var_q_x[:, q].reshape(-1, 1) + 1)
        fraction_term_q = torch.exp(exp_term) / denominator
        # print('fraction_term_q', fraction_term_q)
        fraction_multiply_term = fraction_multiply_term * fraction_term_q
    fi_1 = (output_scale_layer1 ** 2) * fraction_multiply_term
    # print('fi_1',fi_1)


    return fi_1

def fi2_function(N, M, Q, output_scale_layer1, w_q, mean_q_x, var_q_x, Z):
    '''
    fi2 is a M*M matrix
    fi2 = sum^N_n(fi_2_n)
    (fi_2_n)mm' =
    '''
    i = 0
    fi_2 = torch.zeros(M, M).double().to(device)
    for n in range(N):
        # fi_2_n = torch.zeros(M, M).double().to(device)
        fraction_multiply_term = 1
        for q in range(Q):
            z_mq_minus_zmprimeq = torch.cdist(Z[:, q].reshape(-1,1) , Z[:, q].reshape(-1,1) , p=2).pow(2)
            # print('z_mq_minus_zmprimeq',z_mq_minus_zmprimeq)
            # print('z_mq_minus_zmprimeq', z_mq_minus_zmprimeq.shape)
            exp_term1 = - 0.25 * w_q * z_mq_minus_zmprimeq
            Zq = Z[:, q].reshape(-1,1)
            #z_q = 0.5 * (Z[m, q] + Z[m_prime, q])
            z_mq_add_zmprimeq = torch.sum((Zq[:, None] + Zq), dim=-1) *0.5
            # print('z_mq_add_zmprimeq ',z_mq_add_zmprimeq )
            # print('z_mq_add_zmprimeq ', z_mq_add_zmprimeq.shape)

            exp_term2 = - (w_q * ((mean_q_x[n, q] - z_mq_add_zmprimeq) ** 2)) / (
                    2 * w_q * var_q_x[n, q] + 1)
            denominator = torch.sqrt(2 * w_q * var_q_x[n, q] + 1)
            fraction_term_q = torch.exp(exp_term1 + exp_term2) / denominator
            fraction_multiply_term = fraction_multiply_term * fraction_term_q
        fi_2_n= (output_scale_layer1 ** 4) * fraction_multiply_term
        # fi_2_n[m, m_prime] = fi_2_n_mm
        # print('fi_2_n',fi_2_n)

        fi_2 = fi_2 + fi_2_n

    return fi_2

def W_term(N, beta, fi_1, fi_2, K_mm):
    '''
    W = beta *I_N - beta^2 * fi_1 * (beta * fi_2 + K_mm)^(-1) * (fi_1)^T
    '''
    # print('beta',beta)
    # print('fi_1',fi_1)
    W = beta * torch.eye(N).double().to(device) - (beta ** 2) * fi_1 @ torch.inverse(beta * fi_2 + K_mm) @ (fi_1.T)
    # print('torch.inverse(beta * fi_2 + K_mm)',torch.inverse(beta * fi_2 + K_mm))
    return W

def log_term_function(N, beta, fi_2, K_mm, yd, W):
    '''
    log term in loss function
    '''
    log_term1 = 0.5 * N * torch.log(beta) + 0.5 * torch.logdet(K_mm)
    log_term2 = -0.5 * N * torch.log(torch.tensor(2 * math.pi).to(device)) - 0.5 * torch.logdet((beta * fi_2 + K_mm))
    log_exp_term = - 0.5 * yd.T @ W @ yd
    log_term = log_term1 + log_term2 + log_exp_term

    # print('log_term',log_term)
    # print(log_term)
    return log_term

def trace_part_function(beta, fi_0, fi_2, K_mm):
    '''
       trace part in loss function
       trace part  = -(beta * fi_0) / 2 + (bets / 2) * trace(K_mm.inverse @ fi_2)
       '''
    trace_part_1 = -0.5 * beta * fi_0
    # print('K_mm',K_mm)
    trace_part_2 = 0.5 * beta * torch.trace(torch.inverse(K_mm) @ fi_2)
    return trace_part_2 + trace_part_1

def loss_variational_bound(N, M, Q, D, Y, Z,
                           length_scale_layer1, output_scale_layer1, noise_variance,
                           mean_p_x, mean_q_x,
                           var_p_x, log_sigma_q_x,
                           jitter_param
                           ):

    # q(Xq) = N(mean_q(Xq), var_q_x)
    # covar_q_x = torch.zeros(2 * N, 2 * N)
    # covar_q_x[range(2 * N), range(2 * N)] = diag_q_x # size(2N, 2N), covar(qx1) = mean_q_x[:N,:N], covar(qx2) = mean_q_x[N:,N:]

    # std_q_x = torch.nn.functional.softplus(log_sigma_q_x)
    # var_q_x = torch.square(std_q_x)

    var_q_x = log_sigma_q_x
    print('mean_q_x',mean_q_x)
    print('var_q_x',var_q_x)

    mean_vector_Z = torch.zeros(len(Z), 1).double().to(device)
    K_mm = squared_exponential_kernel(
        Z, Z, length_scale_layer1, output_scale_layer1
    ).double()
    print('K_mm',K_mm)
    K_mm += jitter_param * torch.eye(len(Z)).to(device)

    # F(q) = sum_D( Fd(q) ) - KL(q(X)||p(X))
    # kl(q(X)||p(X))
    kl_q_p = KL_divergence(mean_p_x, mean_q_x, var_p_x, var_q_x)
    Fd_q_sum = 0.
    # Fd(q)
    # fi(0), fi(1), fi(2)
    for d in range(D):
        w_q = length_scale_layer1 ** (-2)
        beta = noise_variance ** (-2)
        fi_0 = fi0_function(N, output_scale_layer1)
        fi_1 = fi1_function(N, M, Q, output_scale_layer1, w_q, mean_q_x, var_q_x, Z) # fi_1 is a N*M matrix
        fi_2 = fi2_function(N, M, Q, output_scale_layer1, w_q, mean_q_x, var_q_x, Z) # fi2 is a M*M matrix

        W = W_term(N, beta, fi_1, fi_2, K_mm)
        yd = Y[:, d].reshape(-1, 1)

        log_term = log_term_function(N, beta, fi_2, K_mm, yd, W)
        trace_part = trace_part_function(beta, fi_0, fi_2, K_mm)

        Fd_q = log_term + trace_part
        Fd_q_sum = Fd_q_sum + Fd_q
    F_q = Fd_q_sum - kl_q_p
    # F_q = Fd_q_sum
    return F_q / N

def prediction_function(N, M, Q, X_test,Y, Z,
                        length_scale_layer1_optimize, output_scale_layer1_optimize,
                        w_q, beta, d,
                        mean_q_x_optimize, var_q_x_optimize):
    '''
    m(fd*) = K_n*m (fi_2 + beta^(-1) * Kmm)^(-1) * fi_1^(T) * yd
           = K_n*m * beta *(beta * fi_2 +  Kmm)^(-1) * fi_1^(T) * yd

    cov(fd*) = K_nn_star - K_n*m @ (Kmm^(-1) - (beta * fi_2 + Kmm)^(-1)) @ Kmn*
    '''

    yd = Y[:, d].reshape(-1, 1)
    K_nstar_m = squared_exponential_kernel(
        X_test, Z, length_scale_layer1_optimize, output_scale_layer1_optimize
    ).double()
    K_mm_optimize = squared_exponential_kernel(
        Z, Z, length_scale_layer1_optimize, output_scale_layer1_optimize
    ).double()
    K_nn_star = squared_exponential_kernel(
        X_test, X_test, length_scale_layer1_optimize, output_scale_layer1_optimize
    ).double()
    fi_1 = fi1_function(N, M, Q, output_scale_layer1_optimize, w_q, mean_q_x_optimize, var_q_x_optimize, Z)
    fi_2 = fi2_function(N, M, Q, output_scale_layer1_optimize, w_q, mean_q_x_optimize, var_q_x_optimize, Z)

    # m(fd*)
    # mean_fd_star = K_nstar_m @ torch.inverse(fi_2 + (beta ** (-1)) * K_mm_optimize) @ fi_1.T @ yd
    mean_fd_star = beta * K_nstar_m @ torch.inverse(beta * fi_2 + K_mm_optimize) @ fi_1.T @ yd
    # print('mean_fd_star', mean_fd_star)

    # cov(fd*)
    covar_fd_star = K_nn_star - \
                    K_nstar_m @ (torch.inverse(K_mm_optimize) - torch.inverse(beta * fi_2 + K_mm_optimize)) @ (
                        K_nstar_m.T)
    # print('covar_fd_star', covar_fd_star)
    return mean_fd_star, covar_fd_star



torch.manual_seed(15)

N_star = 0
N = data_points
N_Nstar = N_star + N
Q = 2
D = 1
M = N

# generate data parameter

# output_scale_layer1_generate = torch.tensor([1.])
# length_scale_layer1_generate = torch.tensor([1.5])
# noise_variance_generate = torch.tensor([0.01])

output_scale_layer1_generate = torch.tensor([output_scale_generate]).to(device)
length_scale_layer1_generate = torch.tensor([length_scale_generate]).to(device)
noise_variance_generate = torch.tensor([noise_generate]).to(device)

# model hyperparameter

# output_scale_layer1 = torch.nn.Parameter(torch.tensor([2.], dtype=torch.float64), requires_grad=True)
# length_scale_layer1 = torch.nn.Parameter(torch.tensor([3.], dtype=torch.float64), requires_grad=True)
# noise_variance = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float64), requires_grad=True)

output_scale_layer1 = torch.nn.Parameter(torch.tensor([output_scale_model], dtype=torch.float64).to(device), requires_grad=True)
length_scale_layer1 = torch.nn.Parameter(torch.tensor([length_scale_model], dtype=torch.float64).to(device), requires_grad=True)
noise_variance = torch.nn.Parameter(torch.tensor([noise_model], dtype=torch.float64).to(device), requires_grad=True)

# P(X) = N(0, I)
mean_train_test_X = torch.zeros(N_Nstar, Q).to(device)
var_train_test_X = torch.ones_like(mean_train_test_X).to(device)
std_train_test_X = torch.sqrt(var_train_test_X)
mean_p_x = mean_train_test_X[:N, :].to(device) # size(N,2)
# print('mean_p_x',mean_p_x)
# print('mean_p_x',mean_p_x.shape)
var_p_x = var_train_test_X[:N, :].to(device)
std_p_x = std_train_test_X[:N, :].to(device)
train_test_X_prior = torch.distributions.Normal(mean_train_test_X, std_train_test_X)
samples_train_test_X = train_test_X_prior.sample()
print('samples_p_X',samples_train_test_X)
# mean_sample_train_test_X = samples_train_test_X.mean()
# print('mean_sample_train_test_X',mean_sample_train_test_X)
# Y: random
Y_train_test = train_Y(samples_train_test_X, output_scale_layer1_generate, length_scale_layer1_generate, noise_variance_generate)
Y_train_test = Y_train_test.double()
# print('Y_train_test',Y_train_test)
# print('Y_train_test',Y_train_test.shape)
Y = Y_train_test[:N, :].to(device)
print('Y',Y)

# q(Xq) = N(mean_q(Xq), var_q(Xq))

# mean_q_x = torch.nn.Parameter(torch.randn(N, Q).double().to(device), requires_grad=True) # size(N,2)
# log_sigma_q_x = torch.nn.Parameter(torch.randn(N, Q).double().to(device), requires_grad=True)

mean_q_x = torch.nn.Parameter((torch.ones(N, Q) * 0.1).double().to(device), requires_grad=True) # size(N,2)
log_sigma_q_x = torch.nn.Parameter((torch.ones(N, Q) * 0.1).double().to(device), requires_grad=True)

jitter_param = torch.tensor([0.000001]).double().to(device)

# Z and P(U) = N( m(Z), K1(Z,Z) = K_mm )
p_X_prior = torch.distributions.Normal(mean_p_x, std_p_x)

samples_Z = p_X_prior.sample()
print('samples_Z',samples_Z)
Z = initialize_inducing_inputs(samples_train_test_X, M)
# Z = samples_train_test_X[0:M, :]
# print('Z1',Z)
# Z = samples_train_test_X
Z = Z.double().to(device)
print('Z',Z)

optimizer = optim.Adam([length_scale_layer1, output_scale_layer1, noise_variance, mean_q_x, log_sigma_q_x], lr=learning_rate)

max_grad_norm = 0.5
# num_iterations = 200
loss_set = torch.empty(0).to(device)
length_scale_set = torch.empty(0).to(device)
output_scale_set = torch.empty(0).to(device)
noise_variance_set = torch.empty(0).to(device)
for i in range(num_iterations):
    optimizer.zero_grad()
    loss = -loss_variational_bound(N, M, Q, D, Y, Z,
                           length_scale_layer1, output_scale_layer1, noise_variance,
                           mean_p_x, mean_q_x,
                           var_p_x, log_sigma_q_x,
                           jitter_param
                           )


    print('Iter %d/%d - Loss: %.6f  lengthscale: [%.6f]  outputscale: %.6f  noise: %.6f ' % (
        i + 1, num_iterations, loss.item(),
        length_scale_layer1.item(),
        # length_scale_layer1[1].item(),
        output_scale_layer1.item(),
        noise_variance.item(),
        # jitter_param.item()
    ))


    loss.backward()
    # torch.nn.utils.clip_grad_norm_([output_scale_layer1, length_scale_layer1, noise_variance],
    #                                max_grad_norm)
    optimizer.step()
    print('Iter %d/%d - Loss: %.6f  lengthscale: [%.6f]  outputscale: %.6f  noise: %.6f ' % (
        i + 1, num_iterations, loss.item(),
        length_scale_layer1.item(),
        # length_scale_layer1[1].item(),
        output_scale_layer1.item(),
        noise_variance.item(),
        # jitter_param.item()
    ))
    loss_set = torch.cat((loss_set, torch.tensor([loss.item()]).to(device)))
    length_scale_set = torch.cat((length_scale_set, torch.tensor([length_scale_layer1.item()]).to(device)))
    output_scale_set = torch.cat((output_scale_set, torch.tensor([output_scale_layer1.item()]).to(device)))
    noise_variance_set = torch.cat((noise_variance_set, torch.tensor([noise_variance.item()]).to(device)))

print('       - mean_q_x ',mean_q_x.detach())
print('       - log_sigma_q_x',log_sigma_q_x.detach())
# print('         loss set',loss_set)
# print('         lengthscale set', length_scale_set)
# print('         outputscale set', output_scale_set)
# print('         noise variance set',noise_variance_set)

# prediction
X_test = samples_train_test_X

length_scale_layer1_optimize = length_scale_layer1.detach()
output_scale_layer1_optimize = output_scale_layer1.detach()
noise_variance_optimize = noise_variance.detach()
noise_term = noise_variance_optimize ** (-2)

K_train_train = squared_exponential_kernel(samples_train_test_X, samples_train_test_X, length_scale_layer1_optimize, output_scale_layer1_optimize)
K_train_test = squared_exponential_kernel(samples_train_test_X, X_test,  length_scale_layer1_optimize, output_scale_layer1_optimize)
K_test_test = squared_exponential_kernel(X_test,X_test, length_scale_layer1_optimize,  output_scale_layer1_optimize)

K_train_train += torch.eye(X_test.size(0)).to(device) * noise_variance_optimize ** 2

K_inv = torch.inverse(K_train_train)
mean = K_train_test.t() @ K_inv  @ Y
print('print the predicted mean',mean.reshape(1,-1))
print('Y',Y.reshape(1,-1))
print('X',samples_train_test_X)
print('Z',Z)
print('X',samples_train_test_X.shape)
print('Z',Z.shape)

# nrmse
for task in range(0, D):
    train_rmse = torch.mean(
        torch.pow(mean[:, task] - Y[:, task], 2)).sqrt()
    max_y = torch.max(Y[:, task])
    min_y = torch.min(Y[:, task])
    nrmse = train_rmse / (max_y - min_y)
    print('. NRMSE: %e ' % nrmse)

file_name = f'GPLVM_parameter__data_{N}_Q_{Q}_D_{D}__KMeans_LS{length_scale_generate}_OP{output_scale_generate}_IP{M}_Iter{num_iterations}_LearningR{learning_rate}.pt'
torch.save({'loss_set': loss_set, 'length_scale_set':length_scale_set, 'out_scale_set': output_scale_set, 'noise_variace_set':noise_variance_set}, file_name)




print(f'GPLVM(data_{N}_Q_{Q}_D_{D})_KMeans_LS{length_scale_generate}_OP{output_scale_generate}_IP{M}_Iter{num_iterations}_LearningR{learning_rate}')
X_plot =np.arange(1,num_iterations+1)
# print('X',X)
Y1_f_x = np.array(loss_set.cpu())
# # print(Y1_f_x)
# Y2_error_estimation = np.array(estimation_error_set)
# plt.figure(1)
plt.plot(X_plot,Y1_f_x)
plt.title(f" GPLVM_loss_(data_{N}_Q_{Q}_D_{D})_KMeans_LS{length_scale_generate}_OP{output_scale_generate}_IP{M}_Iter{num_iterations}_LearningR{learning_rate} ")
plt.xlabel('iteration time k')
plt.ylabel('loss')
plt.savefig(f'GPLVM_loss__data_{N}_Q_{Q}_D_{D}__KMeans_LS{length_scale_generate}_OP{output_scale_generate}_IP{M}_Iter{num_iterations}_LearningR{learning_rate}.png')
plt.show()