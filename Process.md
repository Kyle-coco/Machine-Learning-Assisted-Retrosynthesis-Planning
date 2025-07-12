
----------

# EditRetro：逆合成预测模型安装与使用指南

本项目基于论文 **EditRetro: Edit-Based Retrosynthesis Prediction with Documented Edit Sequences**，提出了一种结合模板优势和神经网络能力的逆合成预测方法，广泛用于化学反应路径预测任务。

 **代码来源：**  
[https://zenodo.org/records/11483329](https://zenodo.org/records/11483329)

----------

## 一、环境搭建

### 1. 创建并激活 Conda 环境

打开 Anaconda Prompt，依次执行以下命令：

```bash
conda create -n editretro python=3.10.9
conda activate editretro
```

### 2. 安装 Python 依赖

进入项目根目录，并安装依赖项：

```bash
cd D:\C\AI_Innovation_Practice\ZKD\Practice\yuqianghan-editretro-e954132
pip install -r requirements.txt
```

----------

###  `requirements.txt` 内容示例

```
--extra-index-url https://download.pytorch.org/whl/cu116
numpy==1.23.5
pandas==2.1.4
rdkit==2023.9.2
SmilesPE==0.0.3
textdistance==4.6.0
torch==1.12.0+cu116
tensorboard==2.15.1
```

----------

## 二、安装 Fairseq

Fairseq 是 EditRetro 的核心依赖，需单独安装。

### 1. 安装 Ninja 工具

Fairseq 编译依赖 Ninja：

-   下载地址：[https://github.com/ninja-build/ninja/releases](https://github.com/ninja-build/ninja/releases)
    
-   解压后将 `ninja.exe` 所在路径加入环境变量 `Path`
    
-   或直接安装：
    

```bash
conda install ninja
# 或
pip install ninja
```

测试是否安装成功：

```bash
ninja -v
```

### 2. 设置 CUDA_HOME 环境变量（如使用 GPU）

可在系统环境变量中添加：

```
CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6
```

或在命令行中临时设置：

```bash
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6
```

### 3. 安装 Fairseq

进入 `fairseq` 子目录执行安装：

```bash
cd D:\C\AI_Innovation_Practice\ZKD\Practice\yuqianghan-editretro-e954132\fairseq
pip install --editable ./
```

----------

## 三、常见问题及注意事项

### 1. 未检测到 `cl.exe`（C++ 编译器）

若提示如下错误：

```
UserWarning: Error checking compiler version for cl: [WinError 2] 系统找不到指定的文件。
```

请安装 [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) 并选择：

-    "使用 C++ 的桌面开发（Desktop development with C++）"
    

完成后重启命令行即可。

### 2. Ninja 未添加至 PATH

确保 `ninja.exe` 所在路径已加入环境变量，或使用 `pip/conda` 安装后能直接调用。

----------

## 四、数据预处理

### 1. 下载原始数据集

从以下地址下载原始数据集：

-   **USPTO-50K**：[https://github.com/Hanjun-Dai/GLN](https://github.com/Hanjun-Dai/GLN)（查找 `schneider50k`）
    
-   **USPTO-FULL**：[https://github.com/Hanjun-Dai/GLN](https://github.com/Hanjun-Dai/GLN)（查找 `1976_Sep2016_USPTOgrants_smiles.rsmi` 或 `uspto_multi`）
    

将数据放置于如下路径示例：

```
D:\C\AI_Innovation_Practice\ZKD\Practice\yuqianghan-editretro-e954132\datasets\USPTO_50K\raw
```

### 2. 运行数据预处理脚本

在项目根目录下执行：

```bash
python preprocess_data.py -dataset USPTO_50K -augmentation 1 -processes 4 -spe -dropout 0

python preprocess_data.py -dataset USPTO_FULL -augmentation 5 -processes 8 -spe -dropout 0
```

预处理完成后，结果将保存至：

```
datasets/USPTO_50K/aug1/

datasets/USPTO_FULL/aug5/
```

### 3. 二值化数据

执行以下命令进行二值化：

```bash
sh binarize.sh ./datasets/USPTO_50K/aug1 dict.txt
```

>  **提示：** Windows 默认不支持 `.sh` 脚本，建议使用 [Git Bash](https://git-scm.com/) 运行，或手动复制脚本内容在 PowerShell 中执行（主要包含 `fairseq-preprocess` 命令）。

----------
## 五、模型训练

本项目包含 EditRetro 模型的预训练和微调流程。请确保你已准备好对应的数据集，并且当前路径位于项目根目录：

```
D:\C\AI_Innovation_Practice\ZKD\Practice\yuqianghan-editretro-e954132\datasets\USPTO_50K\raw
```

----------

### 1. 预训练

预训练阶段使用特定的数据集对模型进行初步训练，提升模型的基础能力。

执行预训练的命令：

```bash
sh ./scripts/0_pretrain.sh
```

> **注意：** Windows 用户如果没有安装 Git Bash，需手动执行脚本内命令。

或者你也可以直接使用 `fairseq-train` 命令：

```bash
fairseq-train data-bin/USPTO_50K_aug1 ^
  --user-dir D:/C/AI_Innovation_Practice/ZKD/Practice/yuqianghan-editretro-e954132/editretro ^
  -s src -t tgt ^
  --save-dir results/pretrain_20250709_cpu/checkpoints ^
  --ddp-backend no_c10d ^
  --task translation_pretrain ^
  --criterion pretrain_nat_loss ^
  --arch pretrain_mlm_editretro ^
  --noise random_delete ^
  --share-all-embeddings ^
  --optimizer adam --adam-betas "(0.9,0.98)" ^
  --lr 0.0005 --lr-scheduler inverse_sqrt ^
  --warmup-updates 100 ^
  --warmup-init-lr 1e-07 --label-smoothing 0.1 ^
  --dropout 0.1 --weight-decay 0.01 ^
  --decoder-learned-pos ^
  --encoder-learned-pos ^
  --update-freq 1 ^
  --max-tokens-valid 512 ^
  --distributed-world-size 1 ^
  --log-format simple --log-interval 10 ^
  --fixed-validation-seed 7 ^
  --max-tokens 512 ^
  --save-interval-updates 200 ^
  --max-update 500 ^
  --max-epoch 1 ^
  --keep-last-epochs 1 ^
  --seed 1 ^
  --mask-prob 0.15 ^
  --pretrain

```

![预训练示意图](https://github.com/Kyle-coco/Machine-Learning-Assisted-Retrosynthesis-Planning/blob/2db8eff780e8012ba53784e68d12646f058cd9b6/Pre-training.png)

----------

### 2. 微调

预训练完成后，可以基于下游任务数据进行微调，提高模型在特定任务上的表现。

执行微调的命令：

```bash
sh ./scripts/1_finetune.sh
```

或者直接使用 `fairseq-train` 命令：

```bash
fairseq-train data-bin/USPTO_50K_aug1 ^
  --user-dir editretro ^
  -s src -t tgt ^
  --max-tokens 512 ^
  --max-epoch 1 ^
  --max-update 4000 ^
  --save-interval-updates 1000 ^
  --save-dir results/pretrain_quick ^
  --optimizer adam ^
  --lr 0.0005 ^
  --lr-scheduler inverse_sqrt ^
  --warmup-updates 100 ^
  --clip-norm 1.0 ^
  --dropout 0.1 ^
  --weight-decay 0.0001 ^
  --criterion label_smoothed_cross_entropy ^
  --label-smoothing 0.1 ^
  --no-progress-bar ^
  --log-format simple ^
  --log-interval 100

```

![微调效果图](https://github.com/Kyle-coco/Machine-Learning-Assisted-Retrosynthesis-Planning/blob/2db8eff780e8012ba53784e68d12646f058cd9b6/Model_Fine_Tuning.png)

----------
