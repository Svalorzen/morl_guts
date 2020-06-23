#!/bin/bash

cd momabs/

for SEED in {0..29}
do
python guts.py --bandit-type gaussian --utility-type random_polynomial --objectives 2 --utility-std 0.1 --sig-threshold 0.01 --repeat 1 --horizon 1500 --seeds $SEED
python guts.py --bandit-type gaussian --utility-type random_polynomial --objectives 2 --utility-std 0.5 --sig-threshold 0.01 --repeat 1 --horizon 1500 --seeds $SEED
python guts.py --bandit-type gaussian --utility-type random_polynomial --objectives 2 --utility-std 1.0 --sig-threshold 0.01 --repeat 1 --horizon 1500 --seeds $SEED
python guts.py --bandit-type gaussian --utility-type random_polynomial --objectives 2 --utility-std 0.1 --sig-threshold 1.0  --repeat 1 --horizon 1500 --seeds $SEED
python guts.py --bandit-type gaussian --utility-type random_polynomial --objectives 4 --utility-std 0.1 --sig-threshold 0.01 --repeat 1 --horizon 1500 --seeds $SEED
python guts.py --bandit-type gaussian --utility-type random_polynomial --objectives 4 --utility-std 0.1 --sig-threshold 1.0  --repeat 1 --horizon 1500 --seeds $SEED
python guts.py --bandit-type gaussian --utility-type random_polynomial --objectives 6 --utility-std 0.1 --sig-threshold 0.01 --repeat 1 --horizon 1500 --seeds $SEED
python guts.py --bandit-type gaussian --utility-type random_polynomial --objectives 6 --utility-std 0.1 --sig-threshold 1.0  --repeat 1 --horizon 1500 --seeds $SEED
python guts.py
done

cd ../
mv momabs/runs/ ./
