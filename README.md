## DPGCN

## Train Vanilla GCN

```
python train_gcn.py --dataset cora

```

## Train DPGCN (LapGraph)

```
python train_gcn.py --dataset cora --dp --epsilon 8 --delta 1e-5 --perturb_type continuous --noise_type laplace

```

```
python train_gcn.py --dataset cora --dp --epsilon 8 --delta 1e-5 --perturb_type discrete
```

