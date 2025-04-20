import numpy as np
import mujoco
import time
import mujoco.viewer
import os 
from scipy.spatial.transform import Rotation as R

class KinematicMapper:
    """
    SMPLX到G1映射器，考虑骨骼层级的累积旋转
    """
    
    def __init__(self, robot_model_path, motion_path, lock_waist=False, lock_legs=True):
        """
        初始化映射器
        
        参数:
            robot_model_path: G1机器人模型XML路径
            motion_path: SMPLX运动数据路径
            lock_waist: 是否锁定腰部
            lock_legs: 是否锁定腿部
        """
        # 加载机器人模型
        self.robot_model = mujoco.MjModel.from_xml_path(robot_model_path)
        self.robot_data = mujoco.MjData(self.robot_model)
        self.motion_path = motion_path
        print(f"机器人模型加载成功 - 控制器数量: {self.robot_model.nu}")
        
        # 锁定设置
        self.lock_waist = lock_waist
        self.lock_legs = lock_legs
        print(f"腰部{'锁定' if lock_waist else '未锁定'}")
        print(f"腿部{'锁定' if lock_legs else '未锁定'}")
        
        # 默认角度数组
        self.default_angles = [0.0] * (self.robot_model.nq - 7)
        print("default_angles:", len(self.default_angles))
        
        # 设置默认姿势角度
        default_values = [
            # 左腿 (0-5)
            -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
            # 右腿 (6-11)
            -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
            # 腰部 (12-14)
            -0.2, 0.0, 0.0,
            # 左臂 (15-21)
            0.2, 0.2, 0.0, -0.3, 0.0, 0.0, 0.0,
            # 右臂 (22-28)
            -0.2, 0.2, 0.0, -0.3, 0.0, 0.0, 0.0,
        ]
        
        # 将默认值复制到角度数组
        for i in range(min(len(default_values), len(self.default_angles))):
            self.default_angles[i] = default_values[i]

        self.initial_base_pose = self.robot_data.qpos[:7].copy()
        
        # G1机器人关节ID
        self.g1_joint_ids = {
            # 腰部关节
            "waist_yaw": 13,    # 腰部偏航 (左右转)
            "waist_roll": 14,   # 腰部侧卷 (侧弯)
            "waist_pitch": 15,  # 腰部俯仰 (前后弯)
            
            # 左臂关节
            "left_shoulder_pitch": 16,  # 左肩俯仰
            "left_shoulder_roll": 17,   # 左肩侧卷
            "left_shoulder_yaw": 18,    # 左肩偏航
            "left_elbow": 19,           # 左肘
            
            # 右臂关节
            "right_shoulder_pitch": 23,  # 右肩俯仰
            "right_shoulder_roll": 24,   # 右肩侧卷
            "right_shoulder_yaw": 25,    # 右肩偏航
            "right_elbow": 26,           # 右肘
        }
        
        # SMPLX骨骼的父子关系表 [关节ID, 父关节ID]
        self.smplx_hierarchy = [
            [0, 0],     # 骨盆 (根节点，父节点设为自己)
            [3, 0],     # 脊柱底部, 父节点是骨盆(0)
            [6, 3],     # 脊柱中下部, 父节点是脊柱底部(3)
            [9, 6],     # 脊柱中部/胸部, 父节点是脊柱中下部(6)
            [12, 9],    # 脊柱上部/颈部底部, 父节点是脊柱中部(9)
            [13, 9],    # 左肩膀, 父节点是脊柱中部(9)
            [14, 9],    # 右肩, 父节点是脊柱中部(9)
            [16, 13],   # 左肘, 父节点是左肩(13)
            [17, 14],   # 右肘, 父节点是右肩(14)
            [18, 16],   # 左腕, 父节点是左肘(16)
            [19, 17],   # 右腕, 父节点是右肘(17)
        ]
        
        # 创建父子关系字典，用于快速查找
        self.parent_dict = {joint_id: parent_id for joint_id, parent_id in self.smplx_hierarchy}
        
        # 映射表 {SMPLX关节ID: [(G1关节名, 旋转轴)]}
        self.mapping_table = {
            # 脊柱到腰部映射
            9: [
                ("waist_pitch", 1),  # Y轴：前后弯腰
                ("waist_roll", 0),   # X轴：侧弯腰
                ("waist_yaw", 2)     # Z轴：左右转腰
            ],  
            
            # 左肩膀映射
            13: [
                ("left_shoulder_pitch", 1),  # Y轴：前后抬臂
                ("left_shoulder_roll", 0),   # X轴：侧向抬臂
                ("left_shoulder_yaw", 2)     # Z轴：内外旋转
            ],
            # 左肘映射
            16: [
                ("left_elbow", 1)            # Y轴：肘部弯曲
            ],
            
            # 右肩膀映射
            14: [
                ("right_shoulder_pitch", 1),  # Y轴：前后抬臂
                ("right_shoulder_roll", 0),   # X轴：侧向抬臂
                ("right_shoulder_yaw", 2)     # Z轴：内外旋转
            ],
            # 右肘映射
            17: [
                ("right_elbow", 1)            # Y轴：肘部弯曲
            ],
        }
        
        # 加载运动数据
        self.motion_data = None
        self._load_data()
    
    def _load_data(self):
        """加载SMPLX运动数据"""
        if not os.path.exists(self.motion_path):
            raise FileNotFoundError(f"找不到运动数据文件: {self.motion_path}")
        
        self.motion_data = np.load(self.motion_path)
        print(f"已加载运动数据，总帧数: {self.motion_data['poses'].shape[0]}")
    
    def _extract_local_rotations(self, pose):
        """
        从姿势数据中提取每个关节的局部旋转
        
        参数:
            pose: 姿势数据数组
            
        返回:
            local_rotations: 局部旋转字典 {关节ID: Rotation对象}
        """
        local_rotations = {}
        
        # 为每个关节提取局部旋转
        for joint_id, _ in self.smplx_hierarchy:
            # 获取关节在pose中的索引位置 (每个关节3个值)
            start_idx = joint_id * 3
            if start_idx + 3 <= len(pose):
                # 提取关节旋转向量并转换为旋转对象
                rot_vec = pose[start_idx:start_idx+3]
                local_rotations[joint_id] = R.from_rotvec(rot_vec)
        
        return local_rotations
    
    def _compute_global_rotation(self, joint_id, local_rotations):
        """
        递归计算关节的全局旋转（考虑父关节的累积旋转）
        
        参数:
            joint_id: 关节ID
            local_rotations: 局部旋转字典
            
        返回:
            全局旋转(Rotation对象)
        """
        # 如果是根节点或者关节不在层级中，直接返回局部旋转
        if joint_id == 0 or joint_id not in self.parent_dict or joint_id not in local_rotations:
            return local_rotations.get(joint_id, R.identity())
        
        # 获取父关节ID
        parent_id = self.parent_dict[joint_id]
        
        # 如果父节点是自己，返回局部旋转
        if parent_id == joint_id:
            return local_rotations.get(joint_id, R.identity())
        
        # 获取父关节的全局旋转
        parent_global_rot = self._compute_global_rotation(parent_id, local_rotations)
        
        # 获取当前关节的局部旋转
        local_rot = local_rotations.get(joint_id, R.identity())
        
        # 全局旋转 = 父关节全局旋转 * 当前关节局部旋转
        return parent_global_rot * local_rot
    
    def process_frame(self, frame_index):
        """
        处理单帧SMPLX运动数据，提取关节旋转
        
        返回: 关节旋转字典，{关节ID: Rotation对象}
        """
        # 获取当前帧的姿势数据
        pose = self.motion_data['poses'][frame_index]
        
        # 提取局部旋转
        local_rotations = self._extract_local_rotations(pose)
        
        # 计算每个关节的全局旋转
        global_rotations = {}
        for joint_id in local_rotations.keys():
            global_rotations[joint_id] = self._compute_global_rotation(joint_id, local_rotations)
        
        return global_rotations
    
    def compute_angles(self, joint_rotations):
        """
        根据关节旋转计算G1机器人的关节角度
        
        返回: G1机器人关节角度数组
        """
        # 从默认姿势开始
        angles = np.array(self.default_angles[:self.robot_model.nu])
        
        # 腰部映射
        if not self.lock_waist:
            self._map_torso(angles, joint_rotations)
        
        # 锁定腿部
        if self.lock_legs:
            self._ensure_legs_locked(angles)
        
        # 手臂映射
        self._map_arm(angles, joint_rotations, is_left=True)
        # self._map_arm(angles, joint_rotations, is_left=False)
        
        return angles
    
    def _ensure_legs_locked(self, angles):
        """锁定腿部关节为默认值"""
        # 左腿 (0-5)和右腿 (6-11)
        for i in range(0, 12):
            angles[i] = self.default_angles[i]
    
    def _map_torso(self, angles, joint_rotations):
        """映射脊柱旋转到G1腰部关节"""
        # 使用脊柱胸段(关节9)的旋转
        if 9 in self.mapping_table and 9 in joint_rotations:
            spine_rotation = joint_rotations[9]
            
            # 转换为欧拉角(XYZ顺序)
            euler_angles = spine_rotation.as_euler('xyz', degrees=False)
            
            # 应用映射
            for g1_joint_name, rot_idx in self.mapping_table[9]:
                if g1_joint_name in self.g1_joint_ids:
                    # 获取关节ID和控制器索引
                    joint_id = self.g1_joint_ids[g1_joint_name]
                    ctrl_id = joint_id   # 调整为控制器索引
                    
                    # 获取旋转分量
                    rot_value = euler_angles[rot_idx]
                    
                    # 简单的固定缩放系数
                    scale = 1
                    
                    # 应用角度
                    angles[ctrl_id] = self.default_angles[ctrl_id] + rot_value * scale
    
    def _map_arm(self, angles, joint_rotations, is_left=True):
        """映射手臂旋转到G1关节"""
        # 确定肩关节ID和肘关节ID
        shoulder_id = 13 if is_left else 14
        elbow_id = 16 if is_left else 17
        
        # 映射肩部旋转
        if shoulder_id in self.mapping_table and shoulder_id in joint_rotations:
            shoulder_rotation = joint_rotations[shoulder_id]
            euler_angles = shoulder_rotation.as_euler('xyz', degrees=False)
            
            for g1_joint_name, rot_idx in self.mapping_table[shoulder_id]:
                if g1_joint_name in self.g1_joint_ids:
                    joint_id = self.g1_joint_ids[g1_joint_name]
                    ctrl_id = joint_id   # 调整为控制器索引
                    
                    # 获取旋转分量
                    rot_value = euler_angles[rot_idx]
                    
                    # 简单的固定缩放系数
                    scale = 1
                    
                    # 应用角度
                    angles[ctrl_id] = self.default_angles[ctrl_id] + rot_value * scale
        
        # 映射肘部旋转
        if elbow_id in self.mapping_table and elbow_id in joint_rotations:
            elbow_rotation = joint_rotations[elbow_id]
            euler_angles = elbow_rotation.as_euler('xyz', degrees=False)
            
            for g1_joint_name, rot_idx in self.mapping_table[elbow_id]:
                if g1_joint_name in self.g1_joint_ids:
                    joint_id = self.g1_joint_ids[g1_joint_name]
                    ctrl_id = joint_id   # 调整为控制器索引
                    
                    # 获取旋转分量
                    rot_value = euler_angles[rot_idx]
                    
                    # 简单的固定缩放系数
                    scale = 1
                    
                    # 应用角度
                    angles[ctrl_id] = self.default_angles[ctrl_id] + rot_value * scale
    
    def visualize_kinematic_only(self, start_frame=0, end_frame=None, fps=30):
        """只使用运动学的可视化方法，跳过物理模拟"""
        if end_frame is None:
            end_frame = self.motion_data['poses'].shape[0]
        
        end_frame = min(end_frame, self.motion_data['poses'].shape[0])
        print(f"开始纯运动学可视化 - 从{start_frame}帧到{end_frame}帧")
        
        try:
            with mujoco.viewer.launch_passive(self.robot_model, self.robot_data) as viewer:
                # 循环处理每一帧
                for frame in range(start_frame, end_frame):
                    # 获取关节旋转
                    joint_rotations = self.process_frame(frame)
                    
                    # 计算目标角度
                    target_angles = self.compute_angles(joint_rotations)
                    
                    # 直接设置关节角度 (跳过物理模拟)
                    for i in range(min(len(target_angles), self.robot_model.nu, self.robot_model.nq - 7)):
                        # 设置控制器和qpos
                        self.robot_data.ctrl[i] = target_angles[i]
                        self.robot_data.qpos[i + 7] = target_angles[i]
                    
                    # 保持基座位置固定
                    self.robot_data.qpos[:7] = self.initial_base_pose
                    
                    # 运行前向运动学
                    mujoco.mj_forward(self.robot_model, self.robot_data)
                    
                    # 更新查看器
                    viewer.sync()
                    
                    # 控制帧率
                    time.sleep(1.0/fps)
                    
                    # 显示进度
                    if frame % 50 == 0:
                        print(f"播放进度: {frame}/{end_frame-1} ({frame/(end_frame-1)*100:.1f}%)")
                
                # 动画结束后保持查看器打开
                print("动画播放完毕，查看器保持打开状态")
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.01)
                    
        except Exception as e:
            print(f"可视化过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("可视化结束")

def run_kinematic_demo(robot_model_path, motion_path, lock_waist=False, lock_legs=True):
    """运行纯运动学演示"""
    try:
        print(f"加载运动数据: {motion_path}")
        print(f"加载机器人模型: {robot_model_path}")
        
        # 创建映射器
        mapper = KinematicMapper(robot_model_path, motion_path, lock_waist=lock_waist, lock_legs=lock_legs)
        
        # 获取运动长度
        motion_length = mapper.motion_data['poses'].shape[0]
        print(f"运动数据长度: {motion_length} 帧")
        
        # 运行可视化
        mapper.visualize_kinematic_only(start_frame=0, end_frame=motion_length, fps=30)
            
    except Exception as e:
        print(f"运行演示时发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 文件路径
    motion_path = "/Users/jiafeng/Desktop/cs440_robotic_project/server/output_output.npz"
    robot_path = "/Users/jiafeng/Desktop/cs440_robotic_project/assets/models/robots/g1_robot/classroom_final.xml"
    
    # 运行纯运动学演示
    run_kinematic_demo(robot_path, motion_path, lock_waist=False, lock_legs=True)