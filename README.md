# Dependence
## Install RapidJson

```shell
sudo apt install -y rapidjson-dev 
```

## Install boost

```shell
# 下载boost(>1.78, 此处下载1.82.0)：地址从github Release获取
wget -i https://github.com/boostorg/boost/releases/download/boost-1.82.0/boost-1.82.0.tar.gz
# 安装boost编译依赖
# icu相关依赖给boost:regex使用
sudo apt install -y build-essential g++ python3-dev libicu-dev libbz2-dev
# 解压
tar zxvf boost-1.82.0.tar.gz
cd boost-1.82.0
# boost引导程序设置, 此处设置将boost安装到系统目录
./bootstrap.sh --prefix=/usr/
# 编译
./b2
# 安装
sudo ./b2 install
```

## Install re2

```shell
sudo apt install libre2-dev
```

## Install Nvidia DCGM

地址：[NVIDIA DCGM | NVIDIA Developer](https://developer.nvidia.com/dcgm)

该依赖项仅在ENABLE_GPU_METRIC=True时需要安装

>   依赖项为`dcgm_agent.h`

```shell
# 添加meta-data和gpg key
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"

sudo apt-get update && sudo apt-get install -y datacenter-gpu-manager
```

## Install NUMA

>   依赖项为`numa.h`

```shell
sudo apt-get install libnuma-dev
```

## Install Libevent

参考: https://github.com/libevent/libevent/blob/master/Documentation/Building.md#building-on-unix-cmake

```shell
git clone 

```

## Install Cuda
```
# 更新源
## WSL
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
##  Ubuntu20.04
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
# 安装cuda11.8(pytorch和jax暂时都用这个)
sudo apt-get -y install cuda11.8
```
## Install NCCL

```shell
# 更新源，如果之前安装过cuda可不用做
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
# 安装nccl
sudo apt install libnccl2=2.16.2-1+cuda11.8 libnccl-dev=2.16.2-1+cuda11.8
```
