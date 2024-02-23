import argparse, os, json
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

from samplers import OnlineSampler
from algorithms import Perturbations
from optimizers import SGD

def plot_loss(args):
    
    save = os.path.expanduser(args.output)
    imgDir = os.path.join(save, 'imgs')
    if not os.path.exists(imgDir):
        os.makedirs(imgDir)

    # Load the training data from the JSON file
    rounds = []
    losses = []
    last_round = None
    with open(os.path.join(save, 'train.json'), 'r') as f:
        for line in f:
            data = json.loads(line)
            if last_round is not None and data['round'] != last_round:
                rounds.append(last_round)
                losses.append(last_loss)
            last_round = data['round']
            last_loss = data['loss']
        rounds.append(last_round)
        losses.append(last_loss)

    # Create a new figure
    plt.figure()

    # Plot the loss at the end of each round
    plt.plot(rounds, losses, label='Max Value')

    # Add a legend
    plt.legend()

    # Add labels for the x and y axes
    plt.xlabel('Rounds')
    plt.ylabel('Max Value')
    plt.title('Max Value vs Rounds')

    # Save the figure
    plt.savefig(os.path.join(imgDir, 'rounds_max_plot.png'))

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output", type=str, default='test')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--dataset', type=str, default="linear", choices=['linear', 'multiply'])
    parser.add_argument('--noncvx', type=bool, default=False)
    parser.add_argument("--algo", type=str, default='gs')
    parser.add_argument("--antithetic", type=bool, default=False)
    parser.add_argument("--d", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--L", type=int, default = 20)
    parser.add_argument("--c", type=float, default = 0.1)
    parser.add_argument("--lr", type=float, default = 0.01)
    parser.add_argument("--noise", type=str, default = "no_noise", choices=["no_noise", "fixed_variance"])
    parser.add_argument("--bounds", type=float, nargs=4, default=[0, 1, 0, 1])
    parser.add_argument("--num_data", type=int, default=500)

    ARGS = parser.parse_args()
    arg_dict = vars(ARGS)

    seed = arg_dict["seed"]
    np.random.seed(seed)
    exp_dir = os.getcwd()+'/'+arg_dict["output"]
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    json.dump(arg_dict, open(os.path.join(exp_dir, 'params.json'), 'w'), indent=2, sort_keys=True)

    sampler_linear = OnlineSampler(
            model = arg_dict["dataset"],
            num_data = arg_dict["num_data"],
            xy_bd = arg_dict["bounds"],
            noise = arg_dict["noise"],
            noncvx = arg_dict["noncvx"])

    searcher = Perturbations(
            algo = arg_dict["algo"],
            dim = arg_dict["d"],
            L = arg_dict["L"])

    optimizer = SGD(
            stepsize = arg_dict["lr"])

    # Alternate training and evaluation
    with open(os.path.join(exp_dir, 'train.json'), 'w') as train_progress:
        pass

    for r in range(arg_dict["rounds"]):

        with open(os.path.join(exp_dir, 'train.json'), 'a') as train_progress:

            for i in range(arg_dict["iters"]):
            
                # Generate perturbations
                epsilons = searcher.generate(num_perturbs = arg_dict["L"])

                # Compute pair of estimated risks at each perturbation 
                losses = - sampler_linear.loss(epsilons, arg_dict["c"], antithetic = arg_dict["antithetic"]) # minus sign for finding the maximum

                # Compute gradient estimate
                loss_diff = losses[:,0] - losses[:,1]
                grad = np.dot(np.transpose(epsilons), loss_diff)/(arg_dict["c"]*arg_dict["L"])
                if arg_dict["antithetic"]:
                    grad /= 2

                # Optimizer step
                sampler_linear.params = optimizer.update(sampler_linear.params, grad)

                # Record statistics
                train_progress.write(json.dumps({
                        'round': r,
                        'iteration': i,
                        'loss': -np.mean(losses),
                    }, cls=json.JSONEncoder))
                train_progress.write('\n')

    plot_loss(ARGS)

if __name__ == "__main__":
    main()