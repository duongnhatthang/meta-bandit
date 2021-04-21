# Learning with a small set of optimal arms
- Brief description of the bound & obtained results
- Details of the model architecture and experimental results can be found in our following paper(add link later):

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
 -  Using this (temp)[Notebook](https://github.com/duongnhatthang/meta-bandit/blob/super_clean/main.ipynb)
 -  Using script and check the outputs in `\results`:
        ```
        python eval.py --exp <experiment_type> --noLoadCache <run_new_experiment> \
        --nTasks <int_value> --nArms <int_value> --nExps <int_value> --optSize <int_value> --horizon <int_value> \
        --expArgs <str_value>
        ```
    + `--nTasks`, `--nArms`, `--optSize`, `--horizon`, `--nExps`: the number of tasks, arms, optimal size, horizon and repeated experiments
    + `--expArgs`: arguments for `arms` or `horizon` experiment. Example: `--exp=horizon --expArgs="[100,111,10]"` means doing the horizon experiment with `horizon_list = range(100, 111, 10)`
 -  Examples:
        ```
        python eval.py --exp=task --noLoadCache --nTasks=100 --nArms=5 --nExps=2 --optSize=2 --horizon=100
        python eval.py --exp=horizon --noLoadCache --nTasks=100 --nArms=5 --nExps=2 --optSize=2 --expArgs="[100,111,10]"
        python eval.py --exp=arms --noLoadCache --nTasks=100 --nExps=2 --optSize=2 --horizon=100 --expArgs=[3,5,1]
        python eval.py --exp=subset --noLoadCache --nTasks=100 --nArms=5 --nExps=2 --horizon=100
        ```

## License: Apache 2.0 