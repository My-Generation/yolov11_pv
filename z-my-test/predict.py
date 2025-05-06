#预测
from ultralytics import YOLO

if __name__=='__main__':
    #加载模型，所需参数为模型路径
    model = YOLO(model=r'runs\train\exp\weights\best.pt')
    #执行预测，所需的路径为数据集路径，此处使用自带的图像进行预测
    model.predict(source=r'datasets\BrokenSolarPanelDetection_test\images\test',
                  save=True,
                  show=True)
