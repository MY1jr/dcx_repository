import numpy as np
### numpy方法
## seq
def txt2dataseq(loadfile, time_step=5):
    # 读取txt文件
    raw = np.genfromtxt(loadfile, dtype=None, encoding='utf-8', delimiter=',')
    
    dataseq = []
    # 按标签分组
    current_seq = []
    current_label = None
    
    for row in raw:
        img_path, speed, position = row[0], int(row[1]), int(row[2])
        label = (speed, position)
        
        # 如果是新的标签组或当前序列已满
        if current_label != label or len(current_seq) == time_step:
            # 处理上一组数据(若存在)
            if current_seq:
                # 若不足time_step则用最后一张图片填充
                while len(current_seq) < time_step:
                    current_seq.append(current_seq[-1])
                dataseq.append((current_seq, current_label))
            # 开始新的序列    
            current_seq = [img_path]
            current_label = label
        else:
            current_seq.append(img_path)
    
    # 处理最后一组数据
    if current_seq:
        while len(current_seq) < time_step:
            current_seq.append(current_seq[-1])
        dataseq.append((current_seq, current_label))
        
    return dataseq


