import pandas as pd
from pathlib import Path
from natsort import natsorted

### pandas方法
def data2txt(root_dir, savefile):
    data = []
    root_dir = Path(root_dir)
    for speed_label_dir in root_dir.iterdir():
        if speed_label_dir.is_dir():
            for position_label_dir in speed_label_dir.iterdir():
                if position_label_dir.is_dir():
                    for img_path in position_label_dir.glob('*.[pj][np][ge]*'): # 支持jpg,png,jpeg,bmp,gif,tiff,webp
                        data.append([img_path, int(speed_label_dir.name), int(position_label_dir.name)])
                        print(img_path)
                        
    data = natsorted(data) # 按自然数编号排序  
    df = pd.DataFrame(data, columns=['img_path', 'speed_label', 'position_label'])
    df.to_csv(savefile, index=False, header=False)

    print(f"Saved to: {savefile}")

if __name__ == "__main__":
    current_dir = Path(__file__).parent # utils
    project_root = current_dir.parent.parent # NN_py
    root_dir = project_root / "dcx_cluster"
    txtfile = root_dir / "dcx_cluster.txt"
    #生成txt
    data2txt(root_dir, txtfile)