import sys
sys.path.insert(0,"./")
# 加根目录、子目录，按需加
from pathlib import Path
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "nets"))
sys.path.insert(0, str(root / "nets" / "basicsr"))
from nets.basicsr.models.archs.NAFNet_arch import NAFNet
from ultralytics import YOLO
from ultralytics.nn.modules.restoreBranch import RestoreBranchHasX,RestoreBranchNoX
from ultralytics.nn.modules.classBranch import ClassBranchHasX,ClassBranchNoX

if __name__ == '__main__':
  # Load a model
  model = YOLO("./ultralytics/cfg/models/11/yolo11-segProposedNoX.yaml")  # load a pretrained model (recommended for training)yoloe-11-segOurs
  model = YOLO("./last.pt")  # load a pretrained model (recommended for training)
  # Train the model
  results = model.train(data="./dataset1.yaml", 
                      epochs=150,
                      batch=8,
                      imgsz=1024,
                      save_period=10,
                      scale=0,
                      translate=0,
                      flipud=0.5,
                      fliplr=0.5,
                      patience=300,
                      mask_ratio = 4,
                      device = [1],
                      amp = False,
                      workers = 4,
                      name = "10NoX",
                        )