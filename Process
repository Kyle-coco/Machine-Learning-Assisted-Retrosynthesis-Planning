
# EditRetro：逆合成预测模型安装与使用指南

这份指南将详细说明如何在本地环境（`D:\C\AI_Innovation_Practice\ZKD\Practice\yuqianghan-editretro-e954132`）中设置和运行 EditRetro 项目。

## 一、环境搭建

### 1. 创建并激活 Conda 环境

首先，打开你的 Anaconda Prompt 或 CMD/PowerShell，然后按顺序执行以下命令：

```bash
conda create -n editretro python=3.10.9
conda activate editretro
```

### 2. 安装 Python 依赖

激活环境后，进入你的项目根目录，并安装 `requirements.txt` 中列出的所有依赖：

```bash
cd D:\C\AI_Innovation_Practice\ZKD\Practice\yuqianghan-editretro-e954132
pip install -r requirements.txt
```

----------

# requirements.txt 内容示例

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

这是关键的一步。Fairseq 是 EditRetro 模型的基础。

### 1. 确保已安装 Ninja

Fairseq 的编译过程依赖 Ninja 工具。

-   **Windows 用户安装 Ninja：**
    
    -   你可以从 Ninja 的 [GitHub Release 页面](https://github.com/ninja-build/ninja/releases) 下载预编译的 Windows 版本（`ninja-win.zip`）。
        
    -   解压后，将包含 `ninja.exe` 的目录添加到你的系统环境变量 `Path` 中。
        
    -   或者你可以尝试使用 `conda install ninja` 或 `pip install ninja` 安装。
        
-   **测试 Ninja 是否安装成功：**  
    在命令行输入以下命令，确认能显示版本信息：
   `ninja -v` 
    
### 2. 设置 CUDA_HOME 环境变量

建议在系统环境变量中设置 `CUDA_HOME`，指向你的 CUDA 安装路径。例如：

`CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6` 

你也可以临时在命令行中设置：

`set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6` 

----------

### 3. 进入 fairseq 子目录安装 Fairseq

确保你当前位于 EditRetro 项目的根目录，然后进入 `fairseq` 文件夹执行安装：

`cd D:\C\AI_Innovation_Practice\ZKD\Practice\yuqianghan-editretro-e954132\fairseq`

`pip install --editable ./` 

----------

## 三、常见问题及注意事项

### 1. C++ 编译器缺失（`cl` 命令找不到）

如果你遇到类似如下错误：

`UserWarning:  Error checking compiler version for cl: [WinError 2] 系统找不到指定的文件。` 

这表示系统找不到 Microsoft Visual C++ 编译器 `cl.exe`。

**解决方案：**

-   安装 **Microsoft Visual Studio Build Tools**。
    
-   在安装时，务必勾选“使用 C++ 的桌面开发”（Desktop development with C++）工作负载。
    
-   下载安装链接：  
    [Visual Studio 官方下载页面](https://visualstudio.microsoft.com/downloads/)  
    找到“Tools for Visual Studio 2022”下的“Build Tools”。
    
-   安装完成后，重启命令行窗口或电脑，确保环境变量生效。
    

----------

### 2. Ninja 工具必须正确安装并在 `PATH` 中

尽管报错可能指向编译器，但 Ninja 同样是编译过程中不可缺少的工具。

-   确认 Ninja 可以正常执行（见上文）。
    
-   如果命令行找不到 Ninja，请检查是否正确将其路径添加到系统环境变量中。
-------------------------------------
好的，我帮你整合这部分“二、数据预处理”内容，接在“常见问题及注意事项”之后，形成完整的 README 续写部分，如下：

----------

## 四、数据预处理

在训练模型之前，你需要准备好数据集。

### 1. 下载原始数据集

从原始 README 中提供的链接下载 USPTO-50K 和 USPTO-FULL 数据集：

-   **USPTO-50K:**  
    [https://github.com/Hanjun-Dai/GLN](https://github.com/Hanjun-Dai/GLN) （查找 `schneider50k` 相关文件）
    
-   **USPTO-FULL:**  
    [https://github.com/Hanjun-Dai/GLN](https://github.com/Hanjun-Dai/GLN) （查找 `1976_Sep2016_USPTOgrants_smiles.rsmi` 或 `uspto_multi` 文件夹）
    

下载后，将这些原始数据集文件放入你的项目路径下的相应位置。例如，对于 USPTO-50K：

```
D:\C\AI_Innovation_Practice\ZKD\Practice\yuqianghan-editretro-e954132\datasets\USPTO_50K\raw
```

### 2. 运行数据预处理脚本

确保你当前位于 EditRetro 项目的根目录：

```
D:\C\AI_Innovation_Practice\ZKD\Practice\yuqianghan-editretro-e954132
```

然后执行以下命令来处理数据，处理后的数据将存储在 `datasets/XXX/aug` 文件夹中：

```bash
python preprocess_data.py -dataset USPTO_50K -augmentation 1 -processes 4 -spe -dropout 0

python preprocess_data.py -dataset USPTO_FULL -augmentation 5 -processes 8 -spe -dropout 0
```

### 3. 二值化数据

最后，对处理后的数据进行二值化：

```bash
sh binarize.sh ./datasets/USPTO_50K/aug1 dict.txt
```

> **注意：** Windows 默认不直接支持 `sh` 脚本，你可能需要使用 Git Bash，或者将 `binarize.sh` 中的命令手动复制到 CMD/PowerShell 中执行。通常，它会涉及到 `fairseq-preprocess` 命令。
----------
