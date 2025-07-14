
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
cd D:\C\AI_Innovation_Practice\ZKD\Practice\Editretro_20250712
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
cd D:\C\AI_Innovation_Practice\ZKD\Practice\Editretro_20250712\Fairseq
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
D:\C\AI_Innovation_Practice\ZKD\Practice\Editretro_20250712\datasets\USPTO_50K\raw
```

### 2. 运行数据预处理脚本

在项目根目录下执行：

```bash
python preprocess_data.py -dataset USPTO_50K -augmentation 1 -processes 4 -spe -dropout 0

python preprocess_data.py -dataset USPTO_FULL -augmentation 5 -processes 8 -spe -dropout 0
```

预处理完成后，结果将保存至：

```
D:\C\AI_Innovation_Practice\ZKD\Practice\Editretro_20250712\datasets\USPTO_50K\aug1

D:\C\AI_Innovation_Practice\ZKD\Practice\Editretro_20250712\datasets\USPTO_50K\aug5
```

### 3. 二值化数据

执行以下命令进行二值化：

```bash
sh binarize.sh ./datasets/USPTO_50K/aug1 dict.txt
```
如果卡住不动，执行以下命令：
```
fairseq-preprocess ^
  --source-lang src ^
  --target-lang tgt ^
  --trainpref D:\C\AI_Innovation_Practice\ZKD\Practice\Editretro_20250712\datasets\USPTO_50K\aug1\train ^
  --validpref D:\C\AI_Innovation_Practice\ZKD\Practice\Editretro_20250712\datasets\USPTO_50K\aug1\val ^
  --testpref D:\C\AI_Innovation_Practice\ZKD\Practice\Editretro_20250712\datasets\USPTO_50K\aug1\test ^
  --destdir D:\C\AI_Innovation_Practice\ZKD\Practice\Editretro_20250712\datasets\USPTO_50K\aug1\data-bin\USPTO_50K_aug1 ^
  --srcdict D:\C\AI_Innovation_Practice\ZKD\Practice\Editretro_20250712\preprocess\dict.txt ^
  --tgtdict D:\C\AI_Innovation_Practice\ZKD\Practice\Editretro_20250712\preprocess\dict.txt ^
  --workers 4
```

>  **提示：** Windows 默认不支持 `.sh` 脚本，建议使用 [Git Bash](https://git-scm.com/) 运行，或手动复制脚本内容在 PowerShell 中执行（主要包含 `fairseq-preprocess` 命令）。

----------
## 五、模型训练

本项目包含 EditRetro 模型的预训练和微调流程。请确保你已准备好对应的数据集，并且当前路径位于项目根目录：

项目根目录路径：
```
D:/C/AI_Innovation_Practice/ZKD/Practice/Editretro_20250712
```

数据二值化后的路径：
```
D:/C/AI_Innovation_Practice/ZKD/Practice/Editretro_20250712/datasets/USPTO_50K/aug1/data-bin/USPTO_50K_aug1
```

----------

### 1. 预训练

预训练阶段使用特定的数据集对模型进行初步训练，提升模型的基础能力。

执行预训练的命令：

```bash
bash ./scripts/0_pretrain.sh
```

> **注意：** Windows 用户如果没有安装 Git Bash，需手动执行脚本内命令。

使用脚本前需注意当前目录：
```bash
set PYTHONPATH=D:/C/AI_Innovation_Practice/ZKD/Practice/Editretro_20250712
```

```bash
#!/bin/bash

# 不用显卡训练（CPU-only）
# 请确保使用 (base) 或激活了正确的 conda 环境
# 在 shell 脚本中，即使在 Windows 上，也强烈建议使用正斜杠 / 作为路径分隔符，避免反斜杠 \ 被错误地转义。
databin="D:/C/AI_Innovation_Practice/ZKD/Practice/Editretro_20250712/datasets/USPTO_50K/aug1/data-bin/USPTO_50K_aug1"


noise_type=random_delete
model_args=""  # CPU模式不能使用 --fp16
architecture=pretrain_mlm_editretro
task=translation_pretrain
criterion=pretrain_nat_loss

lr=0.0007
update=4
max_tokens=2048
max_epoch=1
max_update=180

exp_n=pretrain_cpu
root_dir=./results
run_n=$(date "+%Y%m%d_%H%M%S")
exp_dir=$root_dir/$exp_n
mkdir -p $exp_dir

model_dir=${exp_dir}/${run_n}/checkpoints
mkdir -p ${model_dir}

echo "run_n:$run_n, max_tokens:$max_tokens, databin=${databin}, noise_type=${noise_type}, architecture=${architecture}" > $exp_dir/$run_n/config.log
cat $exp_dir/$run_n/config.log

fairseq-train \
    $databin   \
    --user-dir editretro \
    -s src \
    -t tgt \
    --save-dir ${model_dir}  \
    --task ${task}  \
    --criterion ${criterion} \
    --arch ${architecture} \
    --noise ${noise_type} \
    --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9,0.98)' \
    --lr $lr --lr-scheduler inverse_sqrt \
    --warmup-updates 500 \
    --warmup-init-lr '1e-07' --label-smoothing 0.1 \
    --dropout 0.2 --weight-decay 0.01 \
    --decoder-learned-pos \
    --encoder-learned-pos \
    --update-freq ${update} \
    --max-tokens-valid 1000 \
    --log-format 'simple' --log-interval 10 \
    --fixed-validation-seed 7 \
    --max-tokens ${max_tokens} \
    --save-interval-updates 1000 \
    --max-update ${max_update}  \
    --max-epoch ${max_epoch} \
    --keep-last-epochs 10 \
    --seed 1 \
    --mask-prob 0.15 \
    --pretrain \
    ${model_args} > ${model_dir}/pretrain.log 2>&1


```

![预训练结果图]([[https://github.com/Kyle-coco/Machine-Learning-Assisted-Retrosynthesis-Planning/blob/2db8eff780e8012ba53784e68d12646f058cd9b6/Pre-training.png](https://github.com/Kyle-coco/Machine-Learning-Assisted-Retrosynthesis-Planning/blob/main/result_photo/Pre-training.png)](https://github.com/Kyle-coco/Machine-Learning-Assisted-Retrosynthesis-Planning/blob/5576662b50e79d536375b558d7d277b2b6c8af15/result_photo/Pre-training.png))

----------

### 2. 微调

预训练完成后，可以基于下游任务数据进行微调，提高模型在特定任务上的表现。

执行微调的命令：

```bash
bash ./scripts/1_finetune.sh
```

脚本命令如下：

```bash
#!/bin/bash

# ======================= 配置部分 =======================
noise_type=random_delete_shuffle
model_args=""  # CPU 模式不能使用 --fp16
architecture=pretrain_mlm_editretro #editretro_nat
task=translation_retro
criterion=nat_loss

lr=0.00077
update=1
max_tokens=2048
max_epoch=1
max_update=180
log_interval=10

exp_n=finetune_cpu
run_n=$(date "+%Y%m%d_%H%M%S")
root_dir=results
exp_dir=$root_dir/$exp_n
mkdir -p $exp_dir

model_dir=${exp_dir}/${run_n}/checkpoints
mkdir -p ${model_dir}

databin=D:/C/AI_Innovation_Practice/ZKD/Practice/Editretro_20250712/datasets/USPTO_50K/aug1/data-bin/USPTO_50K_aug1
pretrain_ckpt_path=D:/C/AI_Innovation_Practice/ZKD/Practice/Editretro_20250712/results/pretrain_cpu/20250714_091131/checkpoints/checkpoint_best.pt
pretrain_ckpt_name=${pretrain_ckpt_path}


# 保存当前配置
echo "run_n: $run_n" > $exp_dir/$run_n/config.log
echo "databin: $databin" >> $exp_dir/$run_n/config.log
echo "pretrain_ckpt: $pretrain_ckpt_name" >> $exp_dir/$run_n/config.log
echo "lr: $lr, max_tokens: $max_tokens, max_update: $max_update, update_freq: $update" >> $exp_dir/$run_n/config.log
cat $exp_dir/$run_n/config.log

# ======================= 启动微调 =======================
fairseq-train \
  $databin \
  --user-dir "D:/C/AI_Innovation_Practice/ZKD/Practice/Editretro_20250712/editretro" \
  --save-dir $model_dir \
  --task $task \
  --criterion $criterion \
  --arch $architecture \
  --noise $noise_type \
  --share-all-embeddings \
  --optimizer adam --adam-betas '(0.9,0.98)' \
  --lr $lr --lr-scheduler inverse_sqrt \
  --warmup-updates 100 \
  --warmup-init-lr '1e-07' --label-smoothing 0.1 \
  --dropout 0.2 --weight-decay 0.01 \
  --decoder-learned-pos --encoder-learned-pos \
  --max-tokens-valid 1000 \
  --log-format 'simple' --log-interval $log_interval \
  --fixed-validation-seed 7 \
  --max-tokens $max_tokens \
  --max-update $max_update \
  --max-epoch $max_epoch \
  --keep-last-epochs 5 \
  --seed 1 \
  --restore-file ${pretrain_ckpt_name} \
  --reset-optimizer --reset-lr-scheduler --reset-meters --reset-dataloader \
  --pretrain \
  ${model_args} > ${model_dir}/finetune.log 2>&1

```
我拿了微调三个不同阶段的训练结果，以此直观的展现了模型训练的好坏
![微调训练结果图1]([https://github.com/Kyle-coco/Machine-Learning-Assisted-Retrosynthesis-Planning/blob/2db8eff780e8012ba53784e68d12646f058cd9b6/Model_Fine_Tuning.png](https://github.com/Kyle-coco/Machine-Learning-Assisted-Retrosynthesis-Planning/blob/main/result_photo/Finetune3.png))

![微调训练结果图2]([https://github.com/Kyle-coco/Machine-Learning-Assisted-Retrosynthesis-Planning/blob/2db8eff780e8012ba53784e68d12646f058cd9b6/Model_Fine_Tuning.png](https://github.com/Kyle-coco/Machine-Learning-Assisted-Retrosynthesis-Planning/blob/main/result_photo/Finetune2.png))

![微调训练结果图3]([https://github.com/Kyle-coco/Machine-Learning-Assisted-Retrosynthesis-Planning/blob/2db8eff780e8012ba53784e68d12646f058cd9b6/Model_Fine_Tuning.png](https://github.com/Kyle-coco/Machine-Learning-Assisted-Retrosynthesis-Planning/blob/main/result_photo/Finetune.png))

----------
