import numpy as np

# 加载SMPLX模型文件
model_path = "/Users/jiafeng/Desktop/cs440_robotic_project/outside_project/PantoMatrix/datasets/smplx/SMPLX_NEUTRAL_2020.npz"
model_data = np.load(model_path, allow_pickle=True)
print(type(model_data))
# 打印文件中的所有键
print("模型文件中的键:")
for key in model_data.keys():
    if isinstance(model_data[key], np.ndarray):
        print(f"  {key}: 形状 {model_data[key].shape}, 类型 {model_data[key].dtype}")
    else:
        print(f" ........c")

# 检查关节结构
if 'J' in model_data:
    joints = model_data['J']
    print(f"\n关节数量: {len(joints)}")
    print(f"关节数据形状: {joints.shape}")
    
# 检查骨骼结构
if 'kintree_table' in model_data:
    kintree = model_data['kintree_table']
    print(f"\n骨骼层次结构:")
    print(kintree)
    
    # 打印父子关系
    if kintree.shape[0] >= 2:
        print("\n关节父子关系:")
        for i in range(kintree.shape[1]):
            if i > 0:  # 跳过根节点
                parent = kintree[0, i]
                child = kintree[1, i]
                print(f"  关节 {child} 的父关节是 {parent}")

# 加载您的动作数据以比较
motion_path = "/Users/jiafeng/Desktop/cs440_robotic_project/outside_project/PantoMatrix/examples/motion/output_output.npz"
motion_data = np.load(motion_path)
for key in motion_data.keys():
    print(f"  {key}: 形状 {motion_data[key].shape}, 类型 {motion_data[key].dtype}")

# 检查姿态数据
if 'poses' in motion_data:
    poses = motion_data['poses']
    print(f"\n动作数据中的姿态形状: {poses.shape}")
    
    # 如果是3D数组，提取第一帧
    if len(poses.shape) == 3:
        first_frame = poses[0, 0]
    else:
        first_frame = poses[0]
    
    print(f"第一帧姿态数据长度: {len(first_frame)}")
    print(f"如果每个关节3个值，则关节数量: {len(first_frame) // 3}")
    
for i in range(len(first_frame) // 3):
    print(f"  关节 {i} 的旋转角度: {first_frame[3*i:3*i+3]}")