# Non-stationary Bandits and Meta-Learning with a Small Set of Optimal Arms

## Abstract ([RLC 2024 Conference](https://arxiv.org/abs/2202.13001)):
- We study a sequential decision problem where the learner faces a sequence of **K**-armed stochastic bandit tasks. The tasks may be designed by an adversary, but the adversary is constrained to choose the optimal arm of each task in a smaller (but unknown) subset of **M** arms. The task boundaries might be known (the bandit meta-learning setting), or unknown (the non-stationary bandit setting), and the number of tasks **N** as well as the total number of rounds **T** are known (**N** could be unknown in the meta-learning setting). We design an algorithm based on a reduction to bandit submodular maximization, and show that its regret in both settings is smaller than the simple baseline of <img src="https://latex.codecogs.com/svg.image?\tilde{O}(\sqrt{KNT})" title="\tilde{O}(\sqrt{KNT})" /> that can be obtained by using standard algorithms designed for non-stationary bandit problems. For the bandit meta-learning problem with fixed task length <img src="https://latex.codecogs.com/svg.image?\tau" title="\tau" />, we show that the regret of the algorithm is bounded as <img src="https://latex.codecogs.com/svg.image?\tilde{O}(N\sqrt{M&space;\tau}&plus;N^{2/3})" title="\tilde{O}(N\sqrt{M \tau}+N^{2/3})" />. Under additional assumptions on the identifiability of the optimal arms in each task, we show a bandit meta-learning algorithm with an improved <img src="https://latex.codecogs.com/svg.image?\tilde{O}(N\sqrt{M&space;\tau}&plus;N^{1/2})" title="\tilde{O}(N\sqrt{M \tau}+N^{1/2})" /> regret.

## Experiment results:

**Task Experiment**             |  **Small Gap Task Experiment**
:-------------------------:|:-------------------------:
![](https://github.com/duongnhatthang/meta-bandit/blob/main/results/cache_tasks.png)  |  ![](https://github.com/duongnhatthang/meta-bandit/blob/main/results/cache_no_assumption_task.png)
**Subset Experiment**             |  **Small Gap Subset Experiment**
![](https://github.com/duongnhatthang/meta-bandit/blob/main/results/cache_subset.png)  |  ![](https://github.com/duongnhatthang/meta-bandit/blob/main/results/cache_no_assumption_subset.png)

![](https://github.com/duongnhatthang/meta-bandit/blob/main/results/legend5.png)

## Installation 
 -  Python 3.6+

    ```
    git clone https://github.com/duongnhatthang/meta-bandit.git
    cd meta-bandit
    pip3 install -r requirements.txt
    ```

## Evaluation 
 -  [Recommended] Interactive [Notebook](https://github.com/duongnhatthang/meta-bandit/blob/main/main.ipynb), or
 -  Using script and check the outputs in `\results`:

    ```
    python eval.py --exp <experiment_type> --stochastic --adversarial --nonOblivious\
                   --nTasks <int_value> --nArms <int_value> \
                   --nExps <int_value> --optSize <int_value> \
                   --horizon <int_value> --expArgs <str_value> --timeOut <int_value>
    ```

    + `--stochastic/adversarial`, `--nonOblivious`: experiment setting
    + `--nTasks`, `--nArms`, `--optSize`, `--horizon`, `--nExps`: the number of tasks, arms, optimal size, horizon and repeated experiments
    + `--expArgs`: experiment's setting. Example: `--exp=horizon --expArgs="[100,111,10]"` means running the horizon experiment with `horizon_list = range(100, 111, 10)`
    + `--timeOut`: maximum duration of one experiment for each baselines. Total runtime is: timeOut * nExps 
 
 -  Examples:

    ```
    python eval.py --exp=task --stochastic --nArms=5 --nExps=2 --optSize=2 --horizon=100 --expArgs="[100,111,10]"
    python eval.py --exp=horizon --stochastic --nTasks=100 --nArms=5 --nExps=2 --optSize=2 --expArgs="[100,111,10]"
    python eval.py --exp=arms --adversarial --nTasks=100 --nExps=2 --optSize=2 --horizon=100 --expArgs="[3,5,1]"
    python eval.py --exp=subset --adversarial --nTasks=100 --nArms=5 --nExps=2 --horizon=100 --expArgs="[2,6,1]"
    ```

## License: [Apache 2.0](https://github.com/duongnhatthang/meta-bandit/blob/main/LICENSE)
