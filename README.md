# so-vits-svc4.0本地推理 webui个人改良版

创建时间: April 28, 2023 1:11 AM

python环境是3.10.11，因为m1 mac在官方github推荐的python3.8版本下并不能正常使用。本文对m1 mac和linux都有效，Windows理论上也可以，安装步骤可以参照官方github。然后如下操作即可👇

若之前通过我这篇[M1 mac使用so-vits-svc4.0](https://xiaoao.cyou/m1-mac%e4%bd%bf%e7%94%a8so-vits-svc-4-0/)的笔记安装过官网github版本的。可以直接下载我的[github目录](https://github.com/attack-flower/so-vits-svc)中的myapp.py文件和custom_configs文件夹一起丢进本地so-vits-svc根目录，设置好配置，再python myapp.py即可。

预览图：

![https://imgur.xiaoao.cyou/images/2023/04/27/iShot_2023-04-28_02.06.24.png](https://imgur.xiaoao.cyou/images/2023/04/27/iShot_2023-04-28_02.06.24.png)

![https://imgur.xiaoao.cyou/images/2023/04/27/iShot_2023-04-28_02.07.45.png](https://imgur.xiaoao.cyou/images/2023/04/27/iShot_2023-04-28_02.07.45.png)

## 正文

安装好python3.10（mac）/python3.8，或者conda创建一个python3.10的环境。

1. 在终端/命令行，cd到你要放项目的位置，输入以下命令将我的项目下载。

```bash
git clone https://github.com/attack-flower/so-vits-svc.git
```

1. 安装项目所需python依赖包

```bash
pip install -r requirements.txt
```

1. 下载[checkpoint_best_legacy_500.pt](https://ibm.ent.box.com/shared/static/z1wgl1stco8ffooyatzdwsqn2psd9lrr)丢进项目根目录的hubert里。
2. 修改custom_configs里的speakSetting.json文件（使用vscode或在线json文件编辑器去编辑，linux的话可以使用nano/vim修改，windows可以用文本编辑器）

![https://imgur.xiaoao.cyou/images/2023/04/27/iShot_2023-04-28_01.44.00.png](https://imgur.xiaoao.cyou/images/2023/04/27/iShot_2023-04-28_01.44.00.png)

1. 修改完，`python myapp.py`运行即可。

## Docker

第一次编docker，实测我的古董ubuntu能用的。

1. cd到你通常放docker文件的位置。克隆项目。

```bash
git clone https://github.com/attack-flower/so-vits-svc.git
```

1. cd进so-vits-svc文件夹
2. `docker-compose up`等待安装完毕，大概8G左右的流量。
3. 部署完后，请用相应的工具（sftp等）在你的/home/sovits_configs文件夹里放所需的文件（模型、config.json，如果比较多请改好名字以区分。最好新建一个models和configs文件夹来放不然会很乱。），放好后如上面操作修改speakSetting.json即可（路径用“custom_config/你的模型或配置json”）。
4. 最后浏览器输入 ip:7860访问即可。