# centernet3d_onnx_rknn_horizon_tensorRT
CenterNet3D 部署版本，便于移植不同平台（onnx、tensorRT、rknn、Horizon）。


centernet3d_onnx：onnx模型、测试图像、测试结果、测试demo脚本

centernet3d_TensorRT：TensorRT版本模型、测试图像、测试结果、测试demo脚本、onnx模型、onnx2tensorRT脚本(tensorRT-7.2.3.4)

centernet3d_rknn：rknn模型、测试（量化）图像、测试结果、onnx2rknn转换测试脚本

centernet3d_horizon：地平线模型、测试（量化）图像、测试结果、转换测试脚本、测试量化后onnx模型脚本

# 导出onnx

*** 本示例代码只适合按照提供参考导出的onnx，其它导出onn方式自行写后处理 ***  

[导出onnx参考](https://blog.csdn.net/zhangqian_1/article/details/139180009)

# onnx 测试效果

![image](https://github.com/cqu20160901/centernet3d_onnx_rknn_horizon_tensorRT/blob/main/centernet3d_onnx/test_onnx_result.jpg)
