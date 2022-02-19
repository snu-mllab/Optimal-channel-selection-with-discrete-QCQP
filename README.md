# Optimal channel selection with discrete QCQP
This is the code for reproducing the results of the paper Optimal channel selection with discrete QCQP accepted at AISTATS 2022.

## Requirements
- Python 3.7.9
- PyTorch 1.4.0
- TorchVision 0.5.0
- matplotlib 3.3.2
- cplex 12.10.0 => You need to install academic version (restricted usage for community version).

## Download Pretrained Network
- Download 'pretrained_networks.zip' from google drive.
```bash
wget --no-check-certificate -r 'https://docs.google.com/uc?export=download&id=1ZRL84HmoRG28c7IOSDu5gKNwxGcLIsPU' -O pretrained_networks.zip
```
- Unzip to generate './pretrained_networks/'.
```bash
unzip pretrained_networks.zip
```

## Training ResNet on Cifar-10
Run Pruning Training ResNet (depth 56,32,20) on Cifar10:

```bash
python exp/cifar_exp/main.py --mode (mode) --timelimit (timelimit) --baseseed (baseseed) --target_c (target) --arch (arch)
e.g.) python exp/cifar_exp/main.py --mode flops_2D --timelimit 5000 --baseseed 4 --target_c 0.46 --arch resnet20 --use_cplex True
e.g.2) python exp/cifar_exp/main.py --mode flops_2D --timelimit 5000 --baseseed 4 --target_c 0.46 --arch resnet20 --use_cplex False
e.g.3.) python exp/cifar_exp/main.py --mode flops_4D --timelimit 500 --baseseed 4 --target_c 0.46 --gamma 0.5 --arch resnet20 --use_cplex True
e.g.4.) python exp/cifar_exp/main.py --mode flops_4D --timelimit 500 --baseseed 4 --target_c 0.46 --arch resnet20 --use_cplex False
```

* (mode) - 'flops_2D', 'flops_4D', 'mem_2D', 'mem_4D'
* (timelimit) - timelimit for CPLEX. for resnet20 - 5000, resnet32 - 20000, resnet56 - 80000.
* (baseseed) - (4,5,6) for resnet20, (1,2,3) for resnet32, resnet56
* (target_c) - ratio of target constraints and max constraints for conv layers. 0~1
* (gamma) - gamma for 4D. target_c ~ 1
* (arch) - 'resnet20', 'resnet32', 'resnet56'
* (use_cplex) - True or False


### Greedy solve for the discrete QCQP 
We implemented another fast greedy approach to solve discrete QCQP. Please set '(use_cplex)' to be False.
