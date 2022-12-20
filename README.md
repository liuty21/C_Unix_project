# C_Unix_project
## Project
CNN Operational framework
## Filelist
./MNIST：MNIST手写数据集\
./models：按照文本文件格式保存的预训练模型\
&ensp; ├ model_AlexNet\
&ensp; ├ model_ResNet\
&ensp; └ model_VGGNet\
./python：相关的Python代码与模型\
&ensp; ├ load_mnist.py：导入MNIST数据集，生成pytorch格式的Dataset\
&ensp; ├ model_AlexNet.py：AlexNet网络的定义文件\
&ensp; ├ model_ResNet.py：ResNet网络的定义文件\
&ensp; ├ model_VGGNet.py：VGGNet网络的定义文件\
&ensp; ├ model_save.py：将pytorch模型文件保存为文本文件格式，供C++读取\
&ensp; ├ train_model.py：训练pytorch模型的代码\
&ensp; └ *.pth：训练好的pytorch模型文件\
./src：C++卷积神经网络运行框架的代码文件\
&ensp; ├ basic_components.h, basic_components.cpp：基础模块的定义与实现\
&ensp; ├ cblas.h：cblas库的接口定义\
&ensp; ├ CImg.h：CImg图像库文件\
&ensp; ├ libblas.a, libcblas.a：blas与cblas链接库文件\
&ensp; ├ main.cpp：主函数，实现网络推理与结果展示\
&ensp; ├ Makefile文件\
&ensp; └ Models.h, Models.cpp：AlexNet、ResNet与VGGNet网络的定义与实现
