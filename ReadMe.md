# [Dynamic-Meta-Filter-Reproduction](https://github.com/ZYXinnn/Dynamic-Meta-Filter-Reproduction)

## 环境

- Ubuntu 20.04
- RTX 3060 Laptop / A 4000 / A 5000 / M 40 / V 100
- [LibFewShot ](https://github.com/RL-VIG/LibFewShot)& [Dynamic-Meta-filter](https://github.com/chmxu/Dynamic-Meta-filter)
- CUDA == 12.1 / 11.8
- Python 3.9 / 3.11
- GCC/G++ 9.4
- PyTorch


```
- torch
- torchvision
- torchdiffeq
- tqdm
- numpy
- einops
- future
- matplotlib
- numpy
- pandas
- Pillow
- PyYAML
- rich
- scikit-learn
- scipy
- tensorboard
```

## 启动

### 安装框架

#### LibFewShot

```
git clone https://github.com/RL-VIG/LibFewShot.git
```

##### 测试安装

1. 下载 [miniImageNet--ravi.tar.gz](https://box.nju.edu.cn/d/7f6c5bd7cfaf4b019c34/) 并解压到 \your_path\LibFewShot\data\fewshot

2. 修改`run_trainer.py`中`config`设置的一行为

   ```
   config = Config("./config/test_install.yaml").get_config_dict()
   ```

3. 修改`config/headers/data.yaml`中的`data_root`为需要使用的数据集的路径

   ```
   /data/fewshot/miniImageNet--ravi   -->    ./data/fewshot/miniImageNet--ravi
   ```

4. 执行

   ```
   python run_trainer.py
   ```

5. 若第一个epoch输出正常，则表明`LibFewShot`已成功安装。

#### Dynamic-Meta-filter

```
git clone https://github.com/chmxu/Dynamic-Meta-filter.git
```

### 训练与测试

##### 使用Dynamic-Meta-filter源代码

```
python setup.py develop build
python train.py --root {data root} --nExemplars {1/5} --resume {./weights/mini/[1/5]shot.pth.tar}
```

##### 使用LibFewShot

```
cd core
python setup.py develop build
cd /path/to/LibFewShot
python run_trainer.py
```

## 复现结果表

|      | Frame                                                        | Embedding | *mini*ImageNet (5,1) | *mini*ImageNet (5,5) |
| ---- | ------------------------------------------------------------ | --------- | -------------------- | -------------------- |
| 1    | [Dynamic-Meta-filter](https://github.com/chmxu/Dynamic-Meta-filter) | ResNet12  | 67.88 ± 0.49         | 82.11 ± 0.31         |
| 2    | [LibFewShot ](https://github.com/RL-VIG/LibFewShot)          | ResNet12  | 59.453               | ——                   |

具体训练过程见/log
