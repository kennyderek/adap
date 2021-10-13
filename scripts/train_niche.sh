#!/bin/bash

# ablation experiment
python run_farmworld_ablations.py --env-conf ../configs/farmworld/baseline_env.yaml --train-conf ../configs/farmworld/train/adap.yaml --model mult --exp-name ablationAdapMult
sleep 120
python run_farmworld_ablations.py --env-conf ../configs/farmworld/baseline_env.yaml --train-conf ../configs/farmworld/train/diayn.yaml --model mult --exp-name ablationDiaynMult
sleep 120
python run_farmworld_ablations.py --env-conf ../configs/farmworld/baseline_env.yaml --train-conf ../configs/farmworld/train/vanilla.yaml --model mult --exp-name ablationVanillaMult
sleep 120

python run_farmworld_ablations.py --env-conf ../configs/farmworld/baseline_env.yaml --train-conf ../configs/farmworld/train/adap.yaml --model concat --exp-name ablationAdapConcat
sleep 120
python run_farmworld_ablations.py --env-conf ../configs/farmworld/baseline_env.yaml --train-conf ../configs/farmworld/train/diayn.yaml --model concat --exp-name ablationDiaynConcat
sleep 120
python run_farmworld_ablations.py --env-conf ../configs/farmworld/baseline_env.yaml --train-conf ../configs/farmworld/train/vanilla.yaml --model concat --exp-name ablationVanillaConcat

# niche specialization experiment
python run_farmworld_ablations.py --env-conf ../configs/farmworld/niche_specialization_env.yaml --train-conf ../configs/farmworld/train/adap.yaml --model mult --exp-name nicheAdapMult
sleep 120
python run_farmworld_ablations.py --env-conf ../configs/farmworld/niche_specialization_env.yaml --train-conf ../configs/farmworld/train/diayn.yaml --model mult --exp-name nicheDiaynMult
sleep 120
python run_farmworld_ablations.py --env-conf ../configs/farmworld/niche_specialization_env.yaml --train-conf ../configs/farmworld/train/vanilla.yaml --model mult --exp-name nicheVanillaMult
sleep 120

python run_farmworld_ablations.py --env-conf ../configs/farmworld/niche_specialization_env.yaml --train-conf ../configs/farmworld/train/adap.yaml --model concat --exp-name nicheAdapConcat
sleep 120
python run_farmworld_ablations.py --env-conf ../configs/farmworld/niche_specialization_env.yaml --train-conf ../configs/farmworld/train/diayn.yaml --model concat --exp-name nicheDiaynConcat
sleep 120
python run_farmworld_ablations.py --env-conf ../configs/farmworld/niche_specialization_env.yaml --train-conf ../configs/farmworld/train/vanilla.yaml --model concat --exp-name nicheVanillaconcat
