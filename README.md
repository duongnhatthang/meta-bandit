# Bandit Meta-Learning with a Small Set of Optimal Arms

## Abstract:
- We study a meta-learning problem where the learner faces a sequence of **N** multi-armed bandit tasks. Each task is a **K**-armed bandit problem of horizon **T** that may be designed by an adversary, but the adversary is constrained to choose the optimal arm of each task in a smaller (but unknown) subset of **M** arms. 
- We showed an algorithm (G_BASS) with a worst-case regret bounded as <img src="https://latex.codecogs.com/svg.image?\widetilde{O}(N\sqrt{MT}&plus;T\sqrt{KMN})" title="\widetilde{O}(N\sqrt{MT}+T\sqrt{KMN})" />

## Experiment results:

**Task Experiment**             |  **Horizon Experiment**
:-------------------------:|:-------------------------:
![](https://github.com/duongnhatthang/meta-bandit/blob/main/results/cache_tasks.png)  |  ![](https://github.com/duongnhatthang/meta-bandit/blob/main/results/cache_horizon.png)
**Arms Experiment**             |  **Subset Experiment**
![](https://github.com/duongnhatthang/meta-bandit/blob/main/results/cache_arms.png)  |  ![](https://github.com/duongnhatthang/meta-bandit/blob/main/results/cache_subset.png)

Details of the algorithm and the experimental settings can be found in our following paper (update link later):


    @inproceedings{metabandit,
    title     = {{Bandit Meta-Learning with a Small Set of Optimal Arms}},
    author    = {Yasin Abbasi-Yadkori, Thang Duong, Claire Vernade and Andras Gyorgy},
    booktitle = {Update later},
    year      = {2021}
    }

**Please CITE** our paper whenever this repository is used to help produce published results or incorporated into other software.

## Installation 
 -  Python 3.6+

    ```
    git clone (temp)https://github.com/duongnhatthang/meta-bandit.git
    cd meta-bandit
    pip3 install -r requirements.txt
    ```

## Evaluation 
 -  Interactive (temp) [Notebook](https://github.com/duongnhatthang/meta-bandit/blob/main/main.ipynb), or
 -  Using script and check the outputs in `\results`:

    ```
    python eval.py --exp <experiment_type> --notLoadCache <run_new_experiment> \
                   --nTasks <int_value> --nArms <int_value> --nExps <int_value> --optSize <int_value> \
                   --horizon <int_value> --expArgs <str_value>
    ```

    + `--nTasks`, `--nArms`, `--optSize`, `--horizon`, `--nExps`: the number of tasks, arms, optimal size, horizon and repeated experiments
    + `--expArgs`: experiment's setting. Example: `--exp=horizon --expArgs="[100,111,10]"` means running the horizon experiment with `horizon_list = range(100, 111, 10)`
 
 -  Examples:

    ```
    python eval.py --exp=arms --loadCache
    python eval.py --exp=task --notLoadCache --nArms=5 --nExps=2 --optSize=2 --horizon=100 --expArgs="[100,111,10]
    python eval.py --exp=horizon --notLoadCache --nTasks=100 --nArms=5 --nExps=2 --optSize=2 --expArgs="[100,111,10]"
    python eval.py --exp=arms --notLoadCache --nTasks=100 --nExps=2 --optSize=2 --horizon=100 --expArgs=[3,5,1]
    python eval.py --exp=subset --notLoadCache --nTasks=100 --nArms=5 --nExps=2 --horizon=100 --expArgs="[2,6,1]
    ```

## License: [Apache 2.0](https://github.com/duongnhatthang/meta-bandit/blob/main/LICENSE)
