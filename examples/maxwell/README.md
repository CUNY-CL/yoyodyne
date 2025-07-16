# Maxwell edit distance learning

The `transducer` model uses imitation learning to learn to implement a policy based on stochastic edit distance (SED) learned via expectation maximization. The [`maxwell`](https://github.com/CUNY-CL/maxwell) library uses the algorithm of Ristad & Yianilos (1998) to learn this policy from source/target pairs. 

## Usage

The `maxwell-train` command uses a description of TSV files similar to Yoyodyne itself. For the [SIGMORPHON 2016 shared
task](https://sigmorphon.github.io/sharedtasks/2016/) format:

    source  feat1,feat2,... target

the following trains the SED model:

    maxwell-train --target_col 3 --train input.tsv --output sed.pkl

Alternatively, for the [ConLL-SIGMORPHON 2017 shared
task](https://sigmorphon.github.io/sharedtasks/2017/) format:

    source  target  feat1;feat2;...

the following would have a similar effect:

    maxwell-train --train input.tsv --output sed.pkl

Other flags include:

* `--epochs`
* `--source_col`
* `--source_sep`
* `--target_sep`

The path to the resulting `.pkl` file is then provided as an argument to Yoyodyne when training a `TransducerGRU` or `TransducerLSTM` model.

## References

Ristad, E. S. and Yianilos, P. N. 1998. Learning string-edit distance. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ 20(5): 522-532.
