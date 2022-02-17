# Optimal channel selection with discrete QCQP
This is the code for reproducing the results of the paper Optimal channel selection with discrete QCQP accepted at AISTATS 2022.

## Requirements
- Python 3.7.9
- PyTorch 1.4.0
- TorchVision 0.5.0
- matplotlib 3.3.2
- cplex 12.10.0 => You need to install academic version (restricted usage for community version).
- Nimble => Details are provided in the next section (for wall-clock time exp).

## Install Nimble
* Follow installation guide in 'https://github.com/snuspl/nimble/blob/main/NIMBLE_INSTALL.md'.
* After installation, uncomment the following line in **torch/cuda/nimble.py**.
```python
def rewrite_graph(model, dummy_inputs, training=False, use_multi_stream=False):
    if training:
        jit_model = torch.jit.trace(model, dummy_inputs).cuda().train(True)
        torch._C._jit_clear_optimized_graph(jit_model._c)

        prev_autostream_mode = torch._C._cuda_getAutoStreamMode()
        torch._C._cuda_setAutoStreamMode(use_multi_stream)
        jit_model(*dummy_inputs)
        torch._C._cuda_setAutoStreamMode(prev_autostream_mode)

    else:
        # conv selection (only for inference)

        # =====================================================
        # TODO: Uncomment this line
        run_conv_selection(model, dummy_inputs)   
        # =====================================================

        with torch.no_grad(), torch_set_cudnn_enabled(False):
                        jit_model = torch.jit.trace(model, dummy_inputs).cuda().train(False)
                                    # basic operator fusions
                                    torch._C._jit_replace_graph_with_optimized_graph(jit_model._c, *dummy_inputs)
        torch._C._jit_pass_fold_convbn_for_traced_module(jit_model._c)
        torch._C._jit_pass_fold_conv_cat_bn_for_traced_module(jit_model._c)
        torch._C._jit_pass_prepare_elementwise_op_fusion(jit_model._c)

        prev_autostream_mode = torch._C._cuda_getAutoStreamMode()
        torch._C._cuda_setAutoStreamMode(use_multi_stream)
        jit_model(*dummy_inputs)
        torch._C._cuda_setAutoStreamMode(prev_autostream_mode)
    return jit_model
```
* This disables *cudnn* as the backend of Nimble and only enables *CUBLAS* as the backend of Nimble.

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

## ResNet-50 layer Wall-clock time exp

### Step1: Sampling
* Measure the wall-clock time of convolution operations of **Resnet-50** with batch normalization and activation function (relu) while varying the number of input channels (=n_in) and output channels (=n_out). 
```bash
python exp/time_exp/time_layer.py
```
* Add **--use-basic** option for a convolution operation without batch normalization and activation function (relu).
```bash
python exp/time_exp/time_layer.py --use-basic (for basic conv)
```

### Step2: Optimize coefficient
*  Find the best coefficient in quadratic model with samples from Step1 (n_in, n_out, wall clock time) for each convolution operations with batch normalization and activation function (relu).
```bash
python exp/time_exp/solve.py --quad-model (model_type)
```
* (model_type) - 'M5' or 'M6' (See Supplementary material D).
* Add **--use-basic** option for a convolution operation without batch normalization and activation function (relu).
```bash
python exp/time_exp/solve.py --use-basic --quad-model (model_type)
```

### Step3: Evaluation 
* For each samples of pruned **ResNet-50** models, compare the actual wall-clock time and estimated inference time by our quadratic model.
```bash
python exp/time_exp/time_arch.py --quad-model (model_type)
```
* (model_type) - 'M5' or 'M6'
* Add **--use-basic** option for a convolution operation without batch normalization and activation function (relu).
```bash
python exp/time_exp/time_arch.py --use-basic --quad-model (model_type)
```

