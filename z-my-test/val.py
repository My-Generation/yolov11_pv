from ultralytics import YOLO

if __name__ == '__main__':
    #加载训练模型
    model_predict = YOLO(model=r'runs\train\exp\weights\best.pt')
    #进行评估
    metrics = model_predict.val(
        data=r"D:\YOLOV11\ultralytics-8.3.107\datasets\BrokenSolarPanelDetection_test\data.yaml",  # 替换为你的数据集配置文件路径
        split="val",  # 指定验证集
        batch=1,  # 每批次的样本数
        device="0",  # 使用的设备（如 GPU）
        project="runs/val",  # 保存验证结果的项目路径
        name="res",  # 保存验证结果的实验名称
        half=False  # 是否使用半精度浮点数
    )