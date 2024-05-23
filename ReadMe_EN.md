# [Dynamic-Meta-Filter-Reproduction](https://github.com/ZYXinnn/Dynamic-Meta-Filter-Reproduction)

## Getting Started

### Environment

- Ubuntu 22.04

- [LibFewShot](https://github.com/RL-VIG/LibFewShot)

- [CUDA == 12.1](https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local)

- PyTorch == 2.3.0


```
https://download.pytorch.org/whl/torch_stable.html
% pip install https://download.pytorch.org/whl/cu101/torch-1.4.0-cp37-cp37m-win_amd64.whl
% pip install https://download.pytorch.org/whl/cu101/torch-1.7.0%2Bcu101-cp37-cp37m-win_amd64.whl
```

- torchdiffeq == 0.1.1
- torchvision == 0.18.0

```
% pip install https://download.pytorch.org/whl/cu101/torchvision-0.5.0-cp37-cp37m-win_amd64.whl
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

- tqdm

- numpy

### Setup

#### LibFewShot

```
git clone https://github.com/RL-VIG/LibFewShot.git
```

#### Conda

```
conda create -n libfewshot python=3.9
conda activate libfewshot
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torchdiffeq==0.1.1
pip install -r '\your_path\LibFewShot\requirements.txt'
```

### Test the installation

1. download [miniImageNet--ravi.tar.gz](https://box.nju.edu.cn/d/7f6c5bd7cfaf4b019c34/) and extract it to \your_path\LibFewShot\data\fewshot

2. set the `config` as follows in `run_trainer.py`:

   ```
   config = Config("./config/test_install.yaml").get_config_dict()
   ```

3. modify `data_root` in `config/headers/data.yaml` to the path of the dataset to be used.

   ```
   /data/fewshot/miniImageNet--ravi   -->    ./data/fewshot/miniImageNet--ravi
   ```

4. run code

   ```
   python run_trainer.py
   ```

5. If the first output is correct, it means that `LibFewShot` has been successfully installed.

### Port the Dynamic-Meta-filter model into the LibFewShot framework

#### Code format

In the Dynamic-Meta-filter code, ported or used or useless parts use

```python
# use ********************************************
# end_use ****************************************
```

In the LibFewShot framework code, the modified code in `.yaml` is to comment (#) all the original code and rewrite it, and the code added in `.py` is

```python
# add ********************************************
# end_add ****************************************
```

All modified/used code usage

```python
# use_all ********************************************
```

The following is the code that has been completely modified/used

5.22

\LibFewShot\config\headers\optimizer.yaml          ---warmup未引用

\Dynamic-Meta-filter\torchFewShot\optimizers.py

\LibFewShot\config\headers\device.yaml

\LibFewShot\core\model\meta\DynamicWeights.py          ---加入DynamicWeights.py但未完全实现

\LibFewShot\config\headers\losses.yaml

model `DynamicWeightsModel`
- `set_forward`：用于推理阶段调用，返回分类输出以及准确率。
- `set_forward_loss`：用于训练阶段调用，返回分类输出、准确率以及前向损失。
- （`set_forward_adaptation`是微调网络阶段的分类过程所采用的逻辑
- `sub_optimizer`用于在微调时提供新的局部优化器。）

完成lr_scheduler以及warmup:MultiStepLR    ---其中iters_per_epoch: # len(trainloader)  未使用

### Todo
train_loader

