# 苹果手机-越疆机械臂控制
## 简介
该项目介绍了如何在移动端远程语音控制机械臂，项目以苹果手机和越疆机械臂为例，实现简单的语音交互和动作规划，你可以简单修改以支持你的机械臂。

![移动端-客户端-真实机械臂控制](https://github.com/Ghbbbbb/Iphone-dobot/blob/main/assets/framework.png)


## 🛠前置依赖
### 01 物理环境配置
参考dobot官方示例[Dobot物理环境配置](https://github.com/Dobot-Arm/TCP-IP-Python-V3/tree/main)，这边强烈建议你跑通官方的`main.py`后再进行下面的操作。

### 02 移动端配置
安装“快捷指令”软件，然后配置快捷指令，使其可以顺序完成语音转文本和文本存储文件夹两大步骤。配置界面如下图所示，之后需要在PC端配置与Iphone手机的共享文件夹，可以[参考视频](https://www.bilibili.com/video/BV1zk4y167Wi/)。当上述两部配置完成后，可以通过语音输入快捷指令，然后存储到对应文件夹，这时在PC端也会出现对应的文件，支持各类文件的传输，为后续LLM提取文本和结果存储提供了基础。

![移动端配置](https://github.com/Ghbbbbb/Iphone-dobot/blob/main/assets/set.png)

### 03 虚拟环境配置

```
# 克隆项目
git clone https://github.com/Ghbbbbb/Iphone-dobot.git
# 安装依赖包(推荐 python=3.11.x)
pip install -r requirements.txt
``` 

## 🏋️ 运行
- 环境变量设置(windows)
```
#下载embedding模型时如果你访问不了huggingface，设置下面的环境变量
setx HF_ENDPOINT "https://hf-mirror.com"
#设置你的api url和key
setx OPENAI_API_BASE "https://your-proxy-domain/api/v1"
setx OPENAI_API_KEY "your-openai-api-key"
```
- API测试
```
#测试各个子模块是否运行正常
python test.py
```

- LLM生成代码测试
```
#不与服务端进行连接，只测试语言-代码的效果
python client.py --prompt "dobot2" --debug
```

- 机械臂交互和运动
```
#开启服务端
python server.py
#新建一个终端，开启客户端
python client.py --prompt "dobot2"#根据用户键盘输出测试
python client.py --prompt "dobot2" --voice #根据用户语音输入测试，将语音信号上传到`iphone-voice`文件夹
#如果你不想每次都人工输入"y"来执行代码并且你对LLM生成的代码足够确信，你可以在`client.py`(70-72，127-129)中修改成以下断码段：
# print("execute? (y/n):")
# verify = input()
  verify = "y"   
```

在语音交互的模式下，生成的结果将被保存到 `iphone_response` 文件夹下.


## 📈 结果
![Results](https://github.com/Ghbbbbb/Iphone-dobot/blob/main/assets/1.gif)
![Results](https://github.com/Ghbbbbb/Iphone-dobot/blob/main/assets/2.gif)
![Results](https://github.com/Ghbbbbb/Iphone-dobot/blob/main/assets/4.gif)
<div class="item-div">
     <div>
        <img src="https://github.com/Ghbbbbb/Iphone-dobot/blob/main/assets/4.gif"/>
     </div>
</div>
## 📄 联系

如果有任何问题，请[联系我们](http://www.neurcl.cn)