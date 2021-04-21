# Learning with a small set of optimal arms

## Regret bound of the Partial Mornitoring Meta Learning (PMML) algorithm:
- In the agnostic case, with the choice of <a href="https://www.codecogs.com/eqnedit.php?latex=\delta=O\left(\left(\frac{C_{3}^{2}&space;\log&space;Z}{C_{2}^{2}&space;N}\right)^{1&space;/&space;3}\right)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\delta=O\left(\left(\frac{C_{3}^{2}&space;\log&space;Z}{C_{2}^{2}&space;N}\right)^{1&space;/&space;3}\right)" title="\delta=O\left(\left(\frac{C_{3}^{2} \log Z}{C_{2}^{2} N}\right)^{1 / 3}\right)" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=\eta=O\left(\left(\frac{\log&space;^{2}&space;Z}{C_{2}&space;C_{3}^{2}&space;N^{2}}\right)^{1&space;/&space;3}\right)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\eta=O\left(\left(\frac{\log&space;^{2}&space;Z}{C_{2}&space;C_{3}^{2}&space;N^{2}}\right)^{1&space;/&space;3}\right)" title="\eta=O\left(\left(\frac{\log ^{2} Z}{C_{2} C_{3}^{2} N^{2}}\right)^{1 / 3}\right)" /></a>, the regret of the PMML is bounded as <a href="https://www.codecogs.com/eqnedit.php?latex=O\left(\left(C_{2}&space;C_{3}^{2}&space;N^{2}&space;\log&space;Z\right)^{1&space;/&space;3}\right)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?O\left(\left(C_{2}&space;C_{3}^{2}&space;N^{2}&space;\log&space;Z\right)^{1&space;/&space;3}\right)" title="O\left(\left(C_{2} C_{3}^{2} N^{2} \log Z\right)^{1 / 3}\right)" /></a>

- In the realizable case, with the choice of <a href="https://www.codecogs.com/eqnedit.php?latex=\delta=\sqrt{\frac{C_{3}&space;\log&space;Z}{C_{2}&space;N}}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\delta=\sqrt{\frac{C_{3}&space;\log&space;Z}{C_{2}&space;N}}" title="\delta=\sqrt{\frac{C_{3} \log Z}{C_{2} N}}" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=\eta&space;=&space;1" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\eta&space;=&space;1" title="\eta = 1" /></a>, the regret of the algorithm is bounded as <a href="https://www.codecogs.com/eqnedit.php?latex=O\left(\sqrt{C_{2}&space;C_{3}&space;N&space;\log&space;Z}\right)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?O\left(\sqrt{C_{2}&space;C_{3}&space;N&space;\log&space;Z}\right)" title="O\left(\sqrt{C_{2} C_{3} N \log Z}\right)" /></a>

## Experiment results:

**Task Experiment**             |  **Horizon Experiment**
:-------------------------:|:-------------------------:
![](https://github.com/duongnhatthang/meta-bandit/blob/main/results/task_exp_cache.png)  |  ![](https://github.com/duongnhatthang/meta-bandit/blob/main/results/horizon_exp_cache.png)
**Arms Experiment**             |  **Subset Experiment**
![](https://github.com/duongnhatthang/meta-bandit/blob/main/results/arms_exp_cache.png)  |  ![](https://github.com/duongnhatthang/meta-bandit/blob/main/results/subset_exp_cache.png)

Details of the algorithm and the experimental settings can be found in our following paper(add link later):


    @inproceedings{metabandit,
    title     = {{Learning with a small set of optimal arms}},
    author    = {Yasin Abbasi-Yadkori, Thang Duong, Claire Vernade and Andras Gyorgy},
    booktitle = {(temp) Archive ??? },
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
 -  Using this (temp)[Notebook](https://github.com/duongnhatthang/meta-bandit/blob/main/main.ipynb)
 -  Using script and check the outputs in `\results`:

    ```
    python eval.py --exp <experiment_type> --notLoadCache <run_new_experiment> \
                   --nTasks <int_value> --nArms <int_value> --nExps <int_value> --optSize <int_value> \
                   --horizon <int_value> --expArgs <str_value>
    ```

    + `--nTasks`, `--nArms`, `--optSize`, `--horizon`, `--nExps`: the number of tasks, arms, optimal size, horizon and repeated experiments
    + `--expArgs`: arguments for `arms` or `horizon` experiment. Example: `--exp=horizon --expArgs="[100,111,10]"` means doing the horizon experiment with `horizon_list = range(100, 111, 10)`
 
 -  Examples:

    ```
    python eval.py --exp=task --notLoadCache --nTasks=100 --nArms=5 --nExps=2 --optSize=2 --horizon=100
    python eval.py --exp=horizon --notLoadCache --nTasks=100 --nArms=5 --nExps=2 --optSize=2 --expArgs="[100,111,10]"
    python eval.py --exp=arms --notLoadCache --nTasks=100 --nExps=2 --optSize=2 --horizon=100 --expArgs=[3,5,1]
    python eval.py --exp=subset --notLoadCache --nTasks=100 --nArms=5 --nExps=2 --horizon=100
    ```

## License: [Apache 2.0](https://github.com/duongnhatthang/meta-bandit/blob/super_clean/LICENSE)
