# Update these imports at the top of your file
import os
import mujoco
import numpy as np
import time
import cv2
import base64
import re
import threading
import collections
import matplotlib
# Set non-interactive backend to avoid GUI thread issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import sys
import yaml
import torch
from scipy.spatial.transform import Rotation as R



# use abspath to deliminate the ../../ in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


from server.modules.writing_trajectory import  SimpleWritingTrajectory
#from server.modules.labeling_letter import LabelingLetter  


# Modify XML to support 4K resolution
def modify_xml_for_4k_resolution(xml_string):
    """Add or modify visual tags in XML to support 4K framebuffers"""
    # Define the 4K resolution
    width = 4096
    height = 3072
    
    # Check if XML already has visual tags
    if "<visual>" in xml_string:
        # Already has visual tag, check if it has global tag
        if "<global" in xml_string:
            # Use regex to modify existing global tag
            xml_string = re.sub(r'<global([^>]*)/>',
                               f'<global\\1 offwidth="{width}" offheight="{height}"/>',
                               xml_string)
            xml_string = re.sub(r'<global([^>]*?)offwidth="[^"]*"([^>]*?)/>',
                               f'<global\\1offwidth="{width}"\\2/>',
                               xml_string)
            xml_string = re.sub(r'<global([^>]*?)offheight="[^"]*"([^>]*?)/>',
                               f'<global\\1offheight="{height}"\\2/>',
                               xml_string)
        else:
            # Add global tag inside visual tag
            xml_string = xml_string.replace("<visual>", 
                                          f"<visual>\n    <global offwidth=\"{width}\" offheight=\"{height}\"/>")
    else:
        # Add complete visual section after mujoco tag
        xml_string = xml_string.replace("<mujoco", 
                                      f"<mujoco>\n  <visual>\n    <global offwidth=\"{width}\" offheight=\"{height}\"/>\n  </visual>\n", 
                                      1)
        xml_string = xml_string.replace("<mujoco>\n<mujoco>", "<mujoco>", 1)
    
    return xml_string


def find_model_xml(model_name):
    """Search for a model XML file in common locations"""
    possible_paths = [
        f"./{model_name}.xml",
        f"./assets/models/robots/g1_robot/{model_name}.xml", # mac
        os.path.abspath(os.path.join(os.path.dirname(__file__), f"../../assets/models/robots/g1_robot/{model_name}.xml")) # windows
    ]
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found model file at: {path}")
            return path
    return None

class ActionStateManager:
    """Action state controller - manage the action sequence and state transitions"""
    
    def __init__(self, simulator):
        self.simulator = simulator
        self.current_action = None
        self.action_queue = []
        self.is_running = False
        self.socket_emit_func = None
        self.transition_time = 0.5  # action transition time
    
    def set_socket_func(self, socket_emit_func):
        """Set Socket communication function"""
        self.socket_emit_func = socket_emit_func
    
    def queue_actions(self, actions):
        """Set the action queue"""
        self.action_queue = actions.copy()
        self.current_action = None
        return True
    
    def start_sequence(self, socket_emit_func=None):
        """Start executing the action sequence"""
        if socket_emit_func:
            self.set_socket_func(socket_emit_func)
            
        if not self.socket_emit_func:
            print("Warning: Socket emit function not set")
            return False

    
        if self.is_running:
            print("action sequence is already running")
            return False
            
        self.is_running = True
        
        # Start processing thread
        thread = threading.Thread(target=self._process_action_queue)
        thread.daemon = True
        thread.start()
        return True
    
    def _process_action_queue(self):
        """Internal method to process the action queue"""
        try:
            while self.action_queue and self.is_running:
                # Get the next action
                next_action = self.action_queue.pop(0)
                self.current_action = next_action
                
                action_type = next_action.get('type')
                params = next_action.get('params', {})
                
                print(f"\n=== Start executing action: {action_type} ===")
                
                # short stop to ensure previous action has completed
                time.sleep(self.transition_time)
                
                #reset related states
                self._reset_action_states()
                
                # take the action
                success = self._execute_action(action_type, params)
                
                if not success:
                    print(f"action {action_type} execution failed")
                    self.socket_emit_func("action_sequence", {
                        "status": "error", 
                        "action": action_type,
                        "message": "action execution failed"
                    })
                    break
                    
                # waiting for action completion
                self._wait_for_action_completion(action_type)
                
                # action completed
                self.socket_emit_func("action_sequence", {
                    "status": "action_completed", 
                    "action": action_type
                })
                
            # sequence completed
            self.is_running = False
            self.current_action = None
            
            self.socket_emit_func("action_sequence", {
                "status": "sequence_completed"
            })
            
            print("\n=== Action sequence execution complete ===")
            
        except Exception as e:
            print(f"action sequence processing failed: {e}")
            import traceback
            traceback.print_exc()
            
            self.is_running = False
            self.socket_emit_func("action_sequence", {
                "status": "error", 
                "message": str(e)
            })
    
    def _reset_action_states(self):
        """reset all related states"""
        # stop all threaded actions
        self.simulator.walking_in_progress = False
        self.simulator.turning_around = False
        self.simulator.writing_in_progress = False
        self.simulator.sound_with_movement = False
        self.simulator.three_stage_active = False
        
        # make sure thread has enough time to terminate
        time.sleep(0.5)
    
        # reset speed and command
        self.simulator.cmd = np.zeros(3, dtype=np.float32)
        
        # make sure the gesture
        if not self.simulator._is_in_default_pose():
            print("reset tp the default version...")
            self.simulator.reset_to_default_pose_with_interpolation(steps=10)
            
        # release unused resources by using gc module
        import gc
        gc.collect()
    
    def _execute_action(self, action_type, params):
        """execute specified action with parameters"""
        if action_type == 'sound_movement':
            max_frame = params.get('max_frame', 100)
            return self.simulator.sound_with_movement_simulation(self.socket_emit_func, max_frame)
            
        elif action_type == 'turn':
            angle = params.get('angle', 1.57)  # default 90 degrees
            return self.simulator.turning_around_simulation(self.socket_emit_func, [0, 0, angle])
            
        elif action_type == 'stabilize':
            self.simulator.initialize_three_stage_control()
            return True
            
        elif action_type == 'walk':
            distance = params.get('distance', 2.0)
            position = params.get('position', 0)
            if position ==0 :
                return self.simulator.walk_simulation(self.socket_emit_func)
            else:
                return self.simulator.walk_simulation(self.socket_emit_func, [0, distance/4, 0],target_distance=distance)
            
        elif action_type == 'write':
            text = params.get('text', '')
            if not text:
                return False
            # words = text.split()
            return self.simulator.writing_simulation(text, self.socket_emit_func)
            
        else:
            print(f"unknown action type: {action_type}")
            return False
    
    def _wait_for_action_completion(self, action_type):
        """waiting for speficied action to complete"""
        if action_type == 'sound_movement':
            while self.simulator.sound_with_movement:
                time.sleep(0.2)
                
        elif action_type == 'turn':
            while self.simulator.turning_around:
                time.sleep(0.2)
                
        elif action_type == 'stabilize':
            while hasattr(self.simulator, 'three_stage_active') and self.simulator.three_stage_active:
                time.sleep(0.2)
                
        elif action_type == 'walk':
            while self.simulator.walking_in_progress:
                time.sleep(0.2)
                
        elif action_type == 'write':
            while self.simulator.writing_in_progress:
                time.sleep(0.2)
            
        # wait for a small amount of time to ensure the action is completed
        time.sleep(0.5)
        
        

class MujocoSimulator:
    def __init__(self):
        # Initialize variables
        self.simulation_running = False
        self.frame_counter = 0
        self.fps = 0
        self.last_fps_time = time.time()
        self.last_frames = 0
        # Fixed settings
        self.width = 4096
        self.height = 3072
        self.compress_quality = 100
        self.target_fps = 30
        self.last_rendered_frame = None
        self.current_renderer = None
        self.chalk_tip_id = None
        # Shared queue for communication between threads
        self.queue = None
        self.trajectory_points = collections.deque() # create a double-ended queue
        self.trajectory_lock = threading.Lock()
        # Detect best renderer
        self.best_renderer_name = "glfw"
        
        # Model data
        self.model = None  # self.robot_model
        self.data = None   # self.robot_data
        
        self.load_model("classroom_final")
        
        self.writing_in_progress = False
        self.walking_in_progress = False
        self.sound_with_movement = False
        self.turning_around = False
        self.three_stage_active = False
        
        
        # walking parameters
        with open(f"/Users/jiafeng/Desktop/cs440_robotic_project/assets/models/walk/g1_new.yaml", "r", encoding="utf-8") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
            
        # Model paths
        self.policy_path = self.config["policy_path"]        
        # Simulation parameters
        self.simulation_duration = self.config["simulation_duration"]
        self.simulation_dt = self.config["simulation_dt"]
        self.control_decimation = self.config["control_decimation"]
        
        # Control parameters
        self.kps = np.array(self.config["kps"], dtype=np.float32)
        self.kds = np.array(self.config["kds"], dtype=np.float32)
        self.default_angles = np.array(self.config["default_angles"], dtype=np.float32)
        
        # Scaling factors
        self.ang_vel_scale = self.config["ang_vel_scale"]
        self.dof_pos_scale = self.config["dof_pos_scale"]
        self.dof_vel_scale = self.config["dof_vel_scale"]
        self.action_scale = self.config["action_scale"]
        self.cmd_scale = np.array(self.config["cmd_scale"], dtype=np.float32)
        
        # Dimensions
        self.num_actions = self.config["num_actions"]
        self.num_obs = self.config["num_obs"]
        
        # Runtime variables
        self.cmd = [0,0,0]
        self.obs = None
        self.action = None
        self.counter = 0
        self.quat = None
        self.target_dof_pos = None
        self.gravity_orientation = None
        self.policy = None
        self.initial_position = None
        
        
        # sound with movement
        self.lock_waist = False
        self.lock_legs = True
        self.default_angles = [
            # left leg 
            -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
            # right leg
            -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
            # waist 
            -0.2, 0.0, 0.0,  # a little bit bent (z axis adjust)
            # left arm
            0.2, 0.2, 0.0, -0.3, 0.0, 0.0, 0.0,
            # right arm
            -0.2, 0.2, 0.0, -0.3, 0.0, 0.0, 0.0, 
            0.0,
        ]
        
        self.initial_base_pose = self.data.qpos[:7].copy()
        
        # G1 robot joint ID 
        self.g1_joint_ids = {
            # waist joints 
            "waist_yaw": 13,    # 腰部偏航 (左右转)
            "waist_roll": 14,   # 腰部侧卷 (侧弯)
            "waist_pitch": 15,  # 腰部俯仰 (前后弯)
            
            # left arm joints
            "left_shoulder_pitch": 16,  # 左肩俯仰
            "left_shoulder_roll": 17,   # 左肩侧卷
            "left_shoulder_yaw": 18,    # 左肩偏航
            "left_elbow": 19,           # 左肘
            
            # right arm joints
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
        self.motion_path =os.path.abspath(os.path.join(os.path.dirname(__file__), "../output_output.npz"))
        print("motion_path:", self.motion_path)
        self.motion_data = None
        
    
        self.STATE_BRAKING = 0
        self.STATE_STABILIZING = 1 
        self.STATE_COMPLETED = 2
        
        # 当前状态
        self.current_control_state = self.STATE_BRAKING
        self.phase_counter = 0
        
        # 时间参数
        self.braking_frames = int(0.04 / self.simulation_dt)  # 1秒的制动
        self.stabilizing_frames = int(0.04 / self.simulation_dt)  # 1.5秒的稳定
        
        # 保存原始控制参数
        self.original_kps = self.kps.copy() if hasattr(self, 'kps') else None
        self.original_kds = self.kds.copy() if hasattr(self, 'kds') else None
        
        
        
        
        # walking control
        self.distance_threshold = 0.15  # 到达目标的阈值（米）
        self.velocity_threshold = 0.05  # 停止阈值（米/秒）
        self.max_speed = 0.4  # 最大速度命令
        self.min_speed = 0.1  # 最小速度命令
        
        
        
        self.action_manager = ActionStateManager(self)
        
    # 添加高级动作序列方法
    def execute_teaching_sequence(self, socket_emit_func, text_to_speak=""):
        """执行完整教学序列：说话→转向→稳定→走向→写字"""
        
        # 定义动作序列
        action_sequence = [
            {'type': 'sound_movement', 'params': {'max_frame': 100}},
            {'type': 'turn', 'params': {'angle': 3.14/2}},       # 90度转向
            {'type': 'walk', 'params': {'distance': 1.8,'position':0}},     # 走2米
        ]
        for word in text_to_speak:
            if not word.isalpha():
                continue
            for letter in word:
                action_sequence.append({'type': 'write', 'params': {'text': letter}})
                action_sequence.append({'type':'turn','params':{'angle': 1/2}})  # 往左转是正的
                action_sequence.append({'type': 'walk', 'params': {'distance': 0.4, 'position':1}})
                action_sequence.append({'type':'turn','params':{'angle': -1/2}})   
        
        # 设置动作队列并开始执行
        self.action_manager.queue_actions(action_sequence)
        return self.action_manager.start_sequence(socket_emit_func)
        
    def calculate_speed_command(self, remaining_distance):
        """计算基于剩余距离的速度命令 - 简化版"""
        # 简单的距离到速度映射
        if abs(remaining_distance) < 0.5:
            # 接近目标，减速
            speed = max(self.min_speed, abs(remaining_distance) * 0.4)
        else:
            # 正常行走速度
            speed = 0.3
            
        # 确定方向
        direction = 1.0 if remaining_distance > 0 else -1.0
        return speed * direction
        
        
        
    def initialize_three_stage_control(self):
        """初始化三阶段控制参数"""
        print("\n=== 初始化三阶段控制 ===")
        # 确保状态和计数器重置
        self.current_control_state = self.STATE_BRAKING
        self.phase_counter = 0
        
        # 确保缓存原始控制参数
        if not hasattr(self, 'original_kps') or self.original_kps is None:
            self.original_kps = self.kps.copy()
        if not hasattr(self, 'original_kds') or self.original_kds is None:
            self.original_kds = self.kds.copy()
            
        # 标记为活动状态
        self.three_stage_active = True
    
    # movement with action methods
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
        angles = np.array(self.default_angles[:self.model.nu])
        
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
    

    # 修改的三阶段控制函数 - 每帧执行一次
    def three_stage_control(self):
        """按帧执行的三阶段控制系统"""
        # 如果已完成，直接返回
        if not self.three_stage_active:
            self.three_stage_active = False
            return
            
        # 根据当前状态执行对应的控制逻辑
        if self.current_control_state == self.STATE_BRAKING:
            # 制动阶段
            self.phase_counter += 1
            progress = min(1.0, self.phase_counter / self.braking_frames)
            
            # 制动逻辑
            braking_power = (1.0 - progress) * 0.2  # 制动力度随时间减小
            
            # 如果系统支持速度命令
            if hasattr(self, 'cmd'):
                self.cmd[0] = -braking_power
            
            # 修改关节控制增益参数
            for i in range(len(self.kds)):
                self.kds[i] = self.original_kds[i] * (1.0 + progress)  # 增加阻尼
            
            # 检查阶段转换
            if self.phase_counter >= self.braking_frames:
                self.current_control_state = self.STATE_STABILIZING
                self.phase_counter = 0
                print("\n=== 制动完成，开始稳定姿态 ===")
        
        elif self.current_control_state == self.STATE_STABILIZING:
            # 稳定阶段
            self.phase_counter += 1
            progress = min(1.0, self.phase_counter / self.stabilizing_frames)
            
            # 稳定逻辑
            if hasattr(self, 'cmd'):
                self.cmd[0] = 0.0  # 零速度命令
            
            # 修改控制参数
            for i in range(len(self.kps)):
                self.kps[i] = self.original_kps[i] * (1.0 + progress * 0.5)  # 增加刚度
                self.kds[i] = self.original_kds[i] * 1.5  # 保持高阻尼
            
            # 检查阶段转换
            if self.phase_counter >= self.stabilizing_frames:
                self.current_control_state = self.STATE_COMPLETED
                print("\n=== 姿态稳定完成，恢复默认控制参数 ===")
                
                # 恢复原始控制参数
                for i in range(len(self.kps)):
                    self.kps[i] = self.original_kps[i]
                    self.kds[i] = self.original_kds[i]
                
                # 标记控制完成
                self.three_stage_active = False
    
    
    
    
    
    def sound_with_movement_simulation(self, socket_emit_func ,max_frame):
        """
            执行讲话的动作模拟，并将动作数据发送到客户端
        """
        if not self.simulation_running:
            print("Simulation not running, cannot start sound with movement simulation")
            socket_emit_func("simulation_error", {"error": "Simulation not running"})
            return False
        
        if self.writing_in_progress or self.walking_in_progress:
            print("Another action is in progress, cannot start sound with movement")
            socket_emit_func("simulation_error", {"error": "Another action in progress"})
            return False

        self._load_data()
        max_frame = self.motion_data['poses'].shape[0]
        print(f"Start sound with movement simulation for {max_frame} frames")        
        self.sound_with_movement = True
        try:
            writing_thread = threading.Thread(
                target=self._sound_with_movement_thread,
                args=(socket_emit_func, max_frame)
            )
            writing_thread.daemon = True   
            writing_thread.start()         
            return True
        except Exception as e:
            self.sound_with_movement = False
            print(f"Error starting sound with movement simulation: {e}")
            socket_emit_func("simulation_error", {"error": f"Sound with movement error: {str(e)}"})
            return False
    
    
    def _sound_with_movement_thread(self, socket_emit_func, max_frame,fps=30):
        """执行讲话的动作模拟的线程函数"""

        # 加载动作数据
        try:
            # 循环处理每一帧
            for frame in range(0, max_frame):
                # 获取关节旋转
                joint_rotations = self.process_frame(frame)
                # 计算目标角度
                target_angles = self.compute_angles(joint_rotations)
                
                # 直接设置关节角度 (跳过物理模拟)
                for i in range(min(len(target_angles), self.model.nu, self.model.nq - 7)):
                    # 设置控制器和qpos
                    self.data.ctrl[i] = target_angles[i]
                    self.data.qpos[i + 7] = target_angles[i]
                    
                # 保持基座位置固定
                self.data.qpos[:7] = self.initial_base_pose
                
                # 运行前向运动学
                mujoco.mj_forward(self.model, self.data)
                
                
                # 控制帧率
                time.sleep(1.0/fps)
                # 显示进度
                if frame % 50 == 0:
                    progress_value = frame/(max_frame-1)*100
                    print(f"播放进度: {frame}/{max_frame-1} ({progress_value:.1f}%)")
                    socket_emit_func("sound_with_movement_progress", {"progress": f"{progress_value:.1f}"})
            
            self.sound_with_movement = False
            print("Sound with movement simulation finished")
                
        except Exception as e:
            print(f"可视化过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            

            
    def load_model(self, model_name):
        try:
            model_path = find_model_xml(model_name)
            

            if model_path:
                print(f"Found model file: {model_path}")
                # Read XML file - try multiple encodings
                try:
                    # First try UTF-8
                    with open(model_path, 'r', encoding="utf-8") as f:
                        xml_content = f.read()
                    print("Successfully read XML with UTF-8 encoding")
                except UnicodeDecodeError:
                    # If that fails, try latin-1 (can read any byte sequence)
                    with open(model_path, 'r', encoding="latin-1") as f:
                        xml_content = f.read()
                    print("Successfully read XML with latin-1 encoding")
                # Modify XML to support 4K resolution
                modified_xml = modify_xml_for_4k_resolution(xml_content)
                # Load model from modified XML string
                self.model = mujoco.MjModel.from_xml_string(modified_xml)
                
                try:
                    self.chalk_tip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "chalk_tip")
                    print(f"Found chalk_tip site ID: {self.chalk_tip_id}")
                except Exception as e:
                    print(f"Warning: Could not find chalk_tip site: {e}")
                    self.chalk_tip_id = None
                print("Successfully loaded model from modified XML")
            else:
                print(f"Could not find {model_name}.xml, using simple model...")
                raise Exception("Could not find model file")
            self.data = mujoco.MjData(self.model)
            print("Model loaded successfully")
        except Exception as e:
            raise e
            
    
    ####################################################
    # Render frame functions
    ####################################################
    def create_renderer(self):
        """Create a new renderer with fixed 4K resolution"""
        try:
            renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
            renderer.camera = "student_view"
            print(f"Successfully created renderer, size: {self.width}x{self.height}")
            return renderer
        except Exception as e:
            print(f"Failed to create renderer: {e}")
            return None
            
    def render_frame(self):
        """Render a frame with student view camera"""
        try:
            # 确保我们有一个渲染器实例
            if self.current_renderer is None:
                self.current_renderer = self.create_renderer()
                if self.current_renderer is None:
                    raise Exception("Unable to create renderer")
            
            # 创建相机对象并设置学生视角
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat = [0.0, 2.5, 0.6]  # 黑板位置
            camera.distance = 5.0  # 距离
            camera.azimuth = 90.0  # 水平角度
            camera.elevation = -10.0  # 垂直角度
            
            # 使用renderer中已有的_scene和_scene_option
            # 直接访问私有属性可能不是最佳做法，但这取决于renderer的设计
            if hasattr(self.current_renderer, '_scene'):
                # 在更新data前先更新相机参数到场景中
                mujoco.mjv_updateScene(
                    self.model, 
                    self.data,
                    self.current_renderer._scene_option if hasattr(self.current_renderer, '_scene_option') else None,
                    None,  # perturb
                    camera,
                    mujoco.mjtCatBit.mjCAT_ALL.value,
                    self.current_renderer._scene
                )
            
            # 正常渲染流程
            img = self.current_renderer.render()
            img = self.crop_black_borders(img)
            
            # 添加性能信息
            resolution_info = f"{self.width}x{self.height}"
            cv2.putText(
                img, 
                f"FPS: {self.fps:.1f} | {resolution_info}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 255, 255), 
                2
            )
            
            # 缓存此帧
            self.last_rendered_frame = img
            return img
        except Exception as e:
            print(f"Rendering failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def crop_black_borders(self, img, threshold=5):
        """ 
            Crop black borders from the image.
            Pixels with grayscale values ​​below 5 are considered "black".
        """
        if img is None or img.size == 0:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Get non-black boundary
            mask = gray > threshold   #boolean matrix (True / False)   
            rows = np.any(mask, axis=1) #if true , the row is not black
            cols = np.any(mask, axis=0)
            if np.any(rows) and np.any(cols):
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                img = img[rmin:rmax+1, cmin:cmax+1]
            return img
        except Exception as e:
            print(f"Error cropping borders: {e}")
            return img
    
    
    ####################################################
    # General simulation functions (main simulation loop)
    ####################################################
    def unified_simulation_loop(self, socket_emit_func):
        """Unified simulation and visualization loop with thread-safe approach"""
        print("Unified simulation started!")

        # Initialize visualization components
        plt.ioff()  # 关闭 matplotlib 的交互模式，这样图表不会立即显示，而是保存为文件
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        line, = ax.plot([], [], 'r-', linewidth=3)  # Thicker line
        ax.axis('off')
        # Local storage for visualization
        vis_trajectory = []
        
        # Send initial white board
        #创建一个内存缓冲区（文件-like 对象），相当于一个“虚拟文件”，但不会在磁盘写文件
        buf = io.BytesIO()
        fig.savefig(buf, format='jpeg')
        #保存完后，文件指针在末尾，要重新“回到开头”，才能正确读取内容。
        buf.seek(0)
        #把二进制转化成base64(A-Z, a-z, 0-9, +, /)的bytes,在转成Python 字符串 
        frame_data = base64.b64encode(buf.read()).decode('utf-8')
        socket_emit_func("background_frame", {
            "frame": frame_data
        })

        # Main loop
        while self.simulation_running:
            try:
                cycle_start = time.time()
                # 1. Handle physics - only step if writing, otherwise static
                
                if hasattr(self, 'three_stage_active') and self.three_stage_active:
                    print("started three stage control")
                    self.three_stage_control()
                    mujoco.mj_forward(self.model, self.data)
                elif not self.three_stage_active and not self.walking_in_progress and not self.turning_around and not self.writing_in_progress and not self.sound_with_movement:
                    mujoco.mj_forward(self.model, self.data)
                    
                
                # 2. Safely get new points from the deque
                new_points = []
                with self.trajectory_lock:
                    while self.trajectory_points:
                        #从队列左侧（开头）移除并返回一个元素
                        new_points.append(self.trajectory_points.popleft())
                
                # 3. Update local trajectory and visualization if new points
                if new_points:
                    vis_trajectory.extend(new_points)  #将另一个集合中的所有元素添加到列表或队列末尾
                    try:
                        points = np.array(vis_trajectory)
                        line.set_data(points[:, 0], points[:, 1]) # 更新曲线
                        # Save as image
                        buf = io.BytesIO()
                        fig.savefig(buf, format='jpeg')  # 把画好的轨迹画板保存
                        buf.seek(0)
                        # Send blackboard image
                        frame_data = base64.b64encode(buf.read()).decode('utf-8')
                        socket_emit_func("background_frame", {
                            "frame": frame_data
                        })
                    except Exception as e:
                        print(f"Error updating blackboard: {e}")
                        
                # 4. Render 3D scene
                img = self.render_frame()  
                self.frame_counter += 1
                
                # Calculate FPS
                #时间（秒）得到每秒帧数（FPS）
                current_time = time.time()
                elapsed = current_time - self.last_fps_time
                if elapsed >= 1.0:
                    self.fps = (self.frame_counter - self.last_frames) / elapsed
                    self.last_fps_time = current_time
                    self.last_frames = self.frame_counter
                    print(f"Server FPS: {self.fps:.1f} @ {self.width}x{self.height}")

                # Send 3D scene image
                if img is not None:
                    # 1. 将图像编码为 JPEG 格式的二进制数据
                    _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.compress_quality])
                    # 2. 将二进制数据转换为 base64 编码的字符串
                    frame_data = base64.b64encode(buffer).decode("utf-8")
                    socket_emit_func("frame", {
                        "frame": frame_data,
                        "frame_number": self.frame_counter,
                        "server_fps": round(self.fps, 1),
                        "resolution": f"{self.width}x{self.height}"
                    })

                # Control loop timing
                cycle_time = time.time() - cycle_start
                target_cycle_time = 1.0 / self.target_fps
                sleep_time = max(0.001, target_cycle_time - cycle_time)
                time.sleep(sleep_time)
                
            except Exception as e:
                print(f"Simulation error: {e}")
                socket_emit_func("simulation_error", {"error": str(e)})
                time.sleep(0.5)

        # Clean up when simulation stops
        plt.close(fig)
        print("Unified simulation stopped")
        
        
    def _is_in_default_pose(self):
        """检查机器人是否处于默认姿势"""
        default_pose = np.array(self.default_angles[:30])
        current_pose = self.data.qpos[7:37]
        
        # 计算与默认姿势的差异
        diff = np.abs(current_pose - default_pose)
        return np.max(diff) < 0.05  # 允许小的偏差
    
    
    def reset_to_default_pose_with_interpolation(self, steps=20):
        """通过直接设置位置逐步过渡到默认姿势，不使用物理模拟"""
        print("平滑过渡到默认姿势...")
        
        # 保存初始姿势和基座位置
        base_pose = self.data.qpos[:7].copy()
        initial_pose = self.data.qpos[7:37].copy()
        default_pose = np.array(self.default_angles[:30])
        
        # 逐步插值到默认姿势
        for step in range(steps):
            # 计算插值比例
            alpha = (step + 1) / steps
            
            # 计算目标姿势
            target_pose = initial_pose * (1-alpha) + default_pose * alpha
            
            # 直接设置关节角度
            for i in range(min(len(target_pose), self.model.nq - 7)):
                self.data.qpos[i + 7] = target_pose[i]
                self.data.ctrl[i] = target_pose[i]
            
            # 保持基座位置不变
            self.data.qpos[:7] = base_pose
            
            # 清空速度
            self.data.qvel[:] = 0.0
            
            # 更新运动学
            mujoco.mj_kinematics(self.model, self.data)
            
            # 短暂延迟以便可以看到过渡
            time.sleep(0.02)
        
        print("已完成过渡到默认姿势")
    
    
    
    
    def start_simulation(self, socket_emit_func):
        """Start simulation with static robot and visualization in a unified approach"""
        if not self.simulation_running:
            # Set flags
            self.simulation_running = True
            # Clear trajectory points
            with self.trajectory_lock:
                self.trajectory_points.clear()
            # Start a single unified simulation thread
            sim_thread = threading.Thread(target=self.unified_simulation_loop, args=(socket_emit_func,))
            sim_thread.daemon = True
            sim_thread.start()
            print("Unified simulation started")
            return {"success": True, "message": "Simulation started"}
        else:
            return {"success": False, "message": "Simulation already running"}
        
    def stop_simulation(self):
        """Stop all simulation threads"""
        if self.simulation_running:
            self.simulation_running = False
            print("Simulation stopping - cleaning up resources...")
            # Clear trajectory points
            with self.trajectory_lock:
                self.trajectory_points.clear()
            
            
            # Reset flags
            self.writing_in_progress = False
            self.turning_around = False
            self.walking_in_progress = False
            # Wait for threads to terminate
            time.sleep(2.0)
            print("Simulation stopped and resources cleaned up")
            return {"success": True, "message": "Simulation stopped"}
        else:
            print("Simulation is not running")
            return {"success": False, "message": "Simulation is not running"}
    
    
    ####################################################
    # Writing simulation functions
    ####################################################
    def writing_simulation(self, letter, socket_emit_func, retry_count=0, max_retries=2):
        """启动单个字母的写字模拟，带重试逻辑"""
        # 检查模拟是否在运行
        if not self.simulation_running or self.three_stage_active or self.turning_around or self.walking_in_progress:
            print("模拟未运行，无法启动写字")
            socket_emit_func("simulation_error", {"error": "模拟未运行"})
            return False
        
        # 验证并处理字母输入
        if not letter or not isinstance(letter, str) or len(letter.strip()) == 0:
            letter = "B"  # 无效输入时默认使用'A'
        else:
            letter = letter.strip()  # 保留原始大小写
            
        print(f"开始写字模拟: '{letter}'")
        socket_emit_func("writing_status", {"status": "starting", "letter": letter})
        
        # 设置写字状态标志并创建线程
        self.writing_in_progress = True
        try:
            writing_thread = threading.Thread(
                target=self._run_writing_thread,
                args=(letter, socket_emit_func)
            )
            # 一旦线程函数执行完毕，线程就会结束
            writing_thread.daemon = True   #将新创建的线程设置为"守护线程",
                                           #守护线程在主程序退出时会被强制终止，不会等待它们完成
            writing_thread.start()         # 启动线程
            return True
        except Exception as e:
            # 出错时重置标志
            self.writing_in_progress = False
            print(f"启动写字模拟失败: {e}")
            socket_emit_func("simulation_error", {"error": f"写字错误: {str(e)}"})
            return False

    
    def _run_writing_thread(self, letter, socket_emit_func):
        """Thread function for writing with simplified trajectory generation"""
        try:
            print(f"Writing letter '{letter}' on blackboard")
            
            writing = SimpleWritingTrajectory(letter, self.model, self.data, self.chalk_tip_id)
            # Set a better starting position that's visible and centered
            start_pos = np.array([self.data.qpos[0], 4.48, 0.9])
            # Get trajectory for the letter
            letter_trajectory = writing.get_ee_trajectory(start_pos)
            print(f"Will attempt to write {len(letter_trajectory)} points for letter '{letter}'")            
            # Execute each point in the trajectory
            for i, target_pos in enumerate(letter_trajectory):
                if not self.simulation_running:
                    print("Simulation stopped during writing")
                    break
                print(f"Moving to point {i+1}/{len(letter_trajectory)}: {target_pos}")
                # Call our solve_ik method that adds points directly
                success = self.solve_ik_for_position(target_pos)
                if success:
                    socket_emit_func("writing_status", {"status": "progress", "letter": letter, "point": i+1})
                    time.sleep(0.3)  # Increased pause between trajectory points
                else:
                    print(f"Failed to move to position {target_pos}, continuing with next point")
                    time.sleep(0.3)  # Pause even if the move failed
            # Final position feedback
            
            with self.trajectory_lock:
                self.trajectory_points.append([np.nan,np.nan])  # Add a dummy point to indicate movement
            socket_emit_func("writing_status", {"status": "completed", "letter": letter})
            print(f"Writing simulation completed for letter '{letter}'")
            
        except Exception as e:
            print(f"Error in writing simulation: {e}")
            import traceback
            print(traceback.format_exc())
            socket_emit_func("simulation_error", {"error": f"Writing error: {str(e)}"})
        finally:
            # Reset flag and prepare for next operation
            self.writing_in_progress = False
            print(f"Writing thread for letter '{letter}' has finished, system ready for next letter")
    

    
    
    ####################################################
    # Writing helper functions
    ####################################################
    def solve_ik_for_position(self, target_pos, start=23, end=-1):
        """
        修复参数错误的IK求解器，专门优化指定范围的关节（如右手臂）
        """
        try:
            # 检查chalk_tip_id
            if self.chalk_tip_id < 0 or self.chalk_tip_id >= self.model.nsite:
                print("Error: Invalid chalk_tip_id for IK solving")
                return False
                        
            # 获取初始关节角度（只取要优化的部分）
            init_segment = np.copy(self.data.qpos[start:end])
            
            # 定义一个简化IK损失函数
            def ik_loss(joint_angles):
                try:
                    # 创建完整姿势的副本
                    temp_qpos = np.copy(self.data.qpos)
                    # 只更新需要优化的关节段
                    temp_qpos[start:end] = joint_angles
                    # 应用更新后的姿势
                    self.data.qpos[:] = temp_qpos
                    
                    # 计算正向运动学
                    mujoco.mj_kinematics(self.model, self.data)
                    
                    # 计算到目标的距离
                    return np.linalg.norm(self.data.site_xpos[self.chalk_tip_id] - target_pos)
                except Exception as e:
                    print(f"Warning: 在IK损失函数中发生错误: {e}")
                    return 1000.0  # 错误时返回一个大的误差值
            
            # 使用scipy.optimize.minimize
            from scipy.optimize import minimize
            print(f"Solving IK for target position: {target_pos}")
            
            # 注意：minimize使用的初始值和优化参数只包含要优化的关节段
            res = minimize(
                ik_loss,  
                init_segment,  # 只传递需要优化的关节段
                method='SLSQP',
                tol=1e-4,
                options={'maxiter': 100, 'disp': False}
            )
            
            # 处理优化结果
            if res.success:
                final_loss = ik_loss(res.x)
                # 检查NaN或Inf
                if np.isnan(res.x).any() or np.isinf(res.x).any():
                    print("⚠️ 警告: 在IK解中检测到NaN或Inf!")
                    return False
                    
                print(f"IK求解成功，最终误差: {final_loss:.6f}")
                
                # 简单的线性插值，用2步从初始位置到最终位置
                steps = 2
                for alpha in np.linspace(0, 1, steps):
                    # 简单的线性插值
                    interp_segment = init_segment * (1 - alpha) + res.x * alpha
                    
                    # 创建完整姿势的副本，并只更新需要的关节段
                    temp_qpos = np.copy(self.data.qpos)
                    temp_qpos[start:end] = interp_segment
                    self.data.qpos[:] = temp_qpos
                    
                    # 只使用运动学
                    mujoco.mj_kinematics(self.model, self.data)
                    
                    # 获取粉笔尖位置
                    try:
                        tip_pos = np.copy(self.data.site_xpos[self.chalk_tip_id])[[0, 2]]
                        # 归一化坐标用于可视化
                        normalized_x = tip_pos[0] / 2.5
                        normalized_y = (tip_pos[1]) / 2.5
                        # 确保值在[0, 1]范围内用于matplotlib可视化
                        normalized_x = max(0, min(1, normalized_x))
                        normalized_y = max(0, min(1, normalized_y))
                        print(f"Original point: {tip_pos}, Normalized: [{normalized_x:.3f}, {normalized_y:.3f}]")
                        
                        # 使用线程安全方法添加点
                        with self.trajectory_lock:
                            self.trajectory_points.append([normalized_x, normalized_y])
                    except Exception as e:
                        print(f"Warning: 处理尖端位置时出错: {e}")
                        
                    # 控制运动时间
                    time.sleep(0.05)
                return True
            else:
                print(f"IK求解失败: {res.message if hasattr(res, 'message') else '未知优化错误'}")
                return False
        except mujoco.FatalError as e:
            print(f"IK求解过程中发生MuJoCo致命错误: {e}")
            # 恢复数据状态以防止进一步问题
            self.data = mujoco.MjData(self.model)
            return False
        except Exception as e:
            print(f"IK求解过程中发生错误: {e}")
            import traceback
            print(traceback.format_exc())
            return False


    ####################################################
    # Walking simulation functions
    ####################################################
    def turning_around_simulation(self, socket_emit_func, walk_cmd):
        """启动走路模拟，直接使用命令向量"""
        if not self.simulation_running or self.writing_in_progress or self.walking_in_progress or self.three_stage_active:
            print("模拟未运行，无法启动走步")
            socket_emit_func("simulation_error", {"error": "模拟未运行"})
            return False
        
        
        # 设置走路状态标志并创建线程
        self.turning_around = True
        if not self._is_in_default_pose():
            print("Change to the default position")
            self.reset_to_default_pose_with_interpolation()
            
        try:
            turning_around_thread = threading.Thread(
                target=self._turning_around_thread,
                args=(socket_emit_func, walk_cmd)
            )
            turning_around_thread.daemon = True
            turning_around_thread.start()
            time.sleep(0.3)  # 延迟一段时间以便初始化完成
            # self.three_stage_active = True
            return True
        except Exception as e:
            self.turning_around = False
            print(f"启动走步模拟失败: {e}")
            socket_emit_func("simulation_error", {"error": f"走步错误: {str(e)}"})
            return False
    
    def _turning_around_thread(self, socket_emit_func, walk_cmd):
        """执行转向模拟的线程函数 - 基于角度控制"""
        try:
            print(f"Starting turning simulation with command: {walk_cmd}")
            socket_emit_func("walking_status", {"status": "turning", "cmd": walk_cmd})

            # 初始化状态
            self.action = np.zeros(self.num_actions, dtype=np.float32)
            self.target_dof_pos = self.default_angles.copy()
            self.obs = np.zeros(self.num_obs, dtype=np.float32)
            
            # 确保策略加载
            self.model.opt.timestep = self.simulation_dt
            self.policy = torch.jit.load(self.policy_path)
            
            # 设置命令向量和目标
            self.cmd = np.zeros(3, dtype=np.float32)
            self.cmd[:len(walk_cmd)] = walk_cmd  # 复制命令值
            
            # 获取当前四元数和计算当前角度
            current_quat = self.data.qpos[3:7]
            initial_yaw = 2 * np.arctan2(current_quat[3], current_quat[0])
            
            # 计算目标角度（当前角度加上命令角度）
            target_yaw = self.cmd[2]*2 + initial_yaw
            
            # 记录初始位置和角度
            initial_position = self.data.qpos[:2].copy()
            
            # 角度阈值和调整参数
            yaw_threshold = 0.05  # 角度误差阈值（弧度）
            
            print("=== 转向控制信息 ===")
            print(f"初始位置: {initial_position}")
            print(f"初始角度: {initial_yaw:.2f} rad")
            print(f"目标角度: {target_yaw:.2f} rad")
            print(f"角度变化: {self.cmd[2]:.2f} rad")
            
            # 上次状态打印时间
            last_print_time = time.time()
            continuous_stable_frames = 0  # 连续稳定帧计数
            
            # 主转向循环
            while self.simulation_running and self.turning_around:
                step_start = time.time()
                
                # 获取当前状态
                current_position = self.data.qpos[:2]
                current_quat = self.data.qpos[3:7]
                current_yaw = 2 * np.arctan2(current_quat[3], current_quat[0])
                
                # 计算角度误差（考虑角度循环）
                yaw_error = abs((target_yaw - current_yaw + np.pi) % (2 * np.pi) - np.pi)
                
                # 定期状态反馈
                if time.time() - last_print_time > 0.5:
                    if(target_yaw<0):
                        print(f"当前角度: {current_yaw:.2f}, 目标: {target_yaw:.2f}, 误差: {initial_yaw+target_yaw-current_yaw:.3f}")
                    else: 
                        print(f"当前角度: {current_yaw:.2f}, 目标: {target_yaw:.2f}, 误差: {yaw_error:.3f}")
                    last_print_time = time.time()
                
                # 检查是否达到目标角度
                if(target_yaw<0):
                    if initial_yaw+target_yaw-current_yaw < yaw_threshold:
                        continuous_stable_frames += 1
                        # 需要连续多帧稳定才确认完成
                        if continuous_stable_frames >= 10:
                            print("\n=== 转向完成! ===")
                            print(f"最终位置: {current_position}")
                            print(f"最终角度: {current_yaw:.2f}")
                            print(f"角度误差: {yaw_error:.3f}")
                            
                            # 标记转向完成
                            self.turning_around = False
                            socket_emit_func("walking_status", {"status": "turn_completed"})
                            return True
                
                elif yaw_error < yaw_threshold:
                    continuous_stable_frames += 1
                    # 需要连续多帧稳定才确认完成
                    if continuous_stable_frames >= 10:
                        print("\n=== 转向完成! ===")
                        print(f"最终位置: {current_position}")
                        print(f"最终角度: {current_yaw:.2f}")
                        print(f"角度误差: {yaw_error:.3f}")
                        
                        # 标记转向完成
                        self.turning_around = False
                        socket_emit_func("walking_status", {"status": "turn_completed"})
                        return True
                else:
                    continuous_stable_frames = 0  # 重置稳定帧计数
                
                
                if self.current_control_state == self.STATE_BRAKING:
                    # 制动阶段
                    self.phase_counter += 1
                    progress = min(1.0, self.phase_counter / self.braking_frames)
                    
                    # 制动逻辑
                    braking_power = (1.0 - progress) * 0.2  # 制动力度随时间减小
                    
                    # 如果系统支持速度命令
                    if hasattr(self, 'cmd'):
                        self.cmd[0] = -braking_power
                    
                    # 修改关节控制增益参数
                    for i in range(len(self.kds)):
                        self.kds[i] = self.original_kds[i] * (1.0 + progress)  # 增加阻尼
                    
                    # 检查阶段转换
                    if self.phase_counter >= self.braking_frames:
                        self.current_control_state = self.STATE_STABILIZING
                        self.phase_counter = 0
                        print("\n=== 制动完成，开始稳定姿态 ===")
                    

                # 计算控制力矩
                tau = self.pd_control(
                    self.target_dof_pos,
                    self.data.qpos[7:37],
                    self.kps,
                    np.zeros(30),
                    self.data.qvel[6:36],
                    self.kds
                )
                self.data.ctrl[:] = tau
                
                # 应用物理步进
                mujoco.mj_step(self.model, self.data)
                
                # 策略更新
                self.counter += 1
                if self.counter % self.control_decimation == 0:
                    obs_tensor = self.compute_observation()
                    action_12dof = self.policy(obs_tensor).detach().numpy().squeeze()
                    
                    # 扩展动作到所有关节
                    self.action = np.zeros(30, dtype=np.float32)
                    self.action[:12] = action_12dof
                    self.target_dof_pos = self.action * self.action_scale + self.default_angles
                
                # 时间控制
                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
                    
        except Exception as e:
            print(f"转向控制发生错误: {e}")
            import traceback
            traceback.print_exc()
            self.turning_around = False
            socket_emit_func("walking_status", {"status": "error", "message": str(e)})
            return False
    
    def walk_simulation(self, socket_emit_func, walk_cmd=[0.5, 0, 0],target_distance=2.1):
        """启动走路模拟，使用命令向量"""
        if not self.simulation_running or self.writing_in_progress or self.turning_around or self.three_stage_active:
            print("模拟未运行，无法启动走步")
            socket_emit_func("simulation_error", {"error": "模拟未运行"})
            return False

        # 初始化状态
        self.data.qvel[:] = 0.0
        self.data.qpos[7:] = self.default_angles
        time.sleep(0.2)

        self.initial_position = self.data.qpos[:2].copy()
        self.walking_in_progress = True

        try:
            walking_thread = threading.Thread(
                target=self._run_walking_thread,
                args=(socket_emit_func, walk_cmd,target_distance)
            )
            walking_thread.daemon = True
            walking_thread.start()
            time.sleep(0.2)
            return True
        except Exception as e:
            self.walking_in_progress = False
            print(f"Walking simulation failed: {e}")
            socket_emit_func("simulation_error", {"error": f"Walking error: {str(e)}"})
            return False


    def _run_walking_thread(self, socket_emit_func, walk_cmd,target_distance):
        """执行走路模拟的线程函数"""

        print(f"开始走路，命令向量: {walk_cmd}")
        socket_emit_func("walking_status", {"status": "starting", "cmd": walk_cmd})

        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.target_dof_pos = self.default_angles.copy()
        self.obs = np.zeros(self.num_obs, dtype=np.float32)
        self.model.opt.timestep = self.simulation_dt
        self.policy = torch.jit.load(self.policy_path)

        self.cmd[:len(walk_cmd)] = walk_cmd
        self.current_position = self.data.qpos[:2]
        print(f"Robot current position:{self.current_position}")
        self.target_distance = target_distance  # 固定目标距离
        self.distance_threshold = 0.1  # 终止阈值

        print("=== 走路命令信息 ===")
        print(f"初始位置: {self.initial_position}")
        print(f"命令向量: {self.cmd}")

        while self.simulation_running and self.walking_in_progress:
            step_start = time.time()

            current_position = self.data.qpos[:2]
            current_quat = self.data.qpos[3:7]
            # current_yaw = 2 * np.arctan2(current_quat[3], current_quat[0])

            self.current_distance = np.linalg.norm(current_position - self.initial_position)
            remaining_distance = self.target_distance - self.current_distance

            # print(f"当前距离: {self.current_distance:.3f}, 剩余: {remaining_distance:.3f}")

            if remaining_distance < self.distance_threshold:
                print("\n=== 达到目标距离 ===")
                print(f"最终位置: {current_position}")
                print(f"剩余距离: {remaining_distance:.3f}")
                self.walking_in_progress = False
                socket_emit_func("walking_status", {"status": "finished"})
                return

            # 动态更新前进速度
            self.cmd[0] = self.calculate_speed_command(remaining_distance)
            
            if self.current_control_state == self.STATE_BRAKING:
                    # 制动阶段
                    self.phase_counter += 1
                    progress = min(1.0, self.phase_counter / self.braking_frames)
                    
                    # 制动逻辑
                    braking_power = (1.0 - progress) * 0.2  # 制动力度随时间减小
                    
                    # 如果系统支持速度命令
                    if hasattr(self, 'cmd'):
                        self.cmd[0] = -braking_power
                    
                    # 修改关节控制增益参数
                    for i in range(len(self.kds)):
                        self.kds[i] = self.original_kds[i] * (1.0 + progress)  # 增加阻尼
                    
                    # 检查阶段转换
                    if self.phase_counter >= self.braking_frames:
                        self.current_control_state = self.STATE_STABILIZING
                        self.phase_counter = 0
                        print("\n=== 制动完成，开始稳定姿态 ===")

            # PD 控制
            tau = self.pd_control(
                self.target_dof_pos,
                self.data.qpos[7:37],
                self.kps,
                np.zeros(30),
                self.data.qvel[6:36],
                self.kds
            )
            self.data.ctrl[:] = tau
            # 应用控制力矩并推进仿真
            mujoco.mj_step(self.model, self.data)


            # 每 control_decimation 步更新 policy
            self.counter += 1
            if self.counter % self.control_decimation == 0:
                obs_tensor = self.compute_observation()
                action_12dof = self.policy(obs_tensor).detach().numpy().squeeze()
                self.action = np.zeros(30, dtype=np.float32)
                self.action[:12] = action_12dof
                self.target_dof_pos = self.action * self.action_scale + self.default_angles

            # keep the time sime 
            time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

                    

    
    ####################################################
    # Walking helper functions
    ####################################################

    def get_gravity_orientation(self, quaternion):
        """Calculate gravity orientation from quaternion"""
        qw, qx, qy, qz = quaternion
        
        gravity_orientation = np.zeros(3)
        gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
        gravity_orientation[1] = -2 * (qz * qy + qw * qx)
        gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
        
        return gravity_orientation

    def pd_control(self, target_pos, current_pos, kp, target_vel, current_vel, kd):
        """Calculate torques using PD control
            The PD controller is responsible for smoothly moving the 
            joints to these continuously updated target positions.
            
            Args:
                target_pos: Target joint positions
                current_pos: Current joint positions
                kp: Position gains
                target_vel: Target joint velocities
                current_vel: Current joint velocities
                kd: Velocity gains
        """
        return (target_pos - current_pos) * kp + (target_vel - current_vel) * kd

    def compute_observation(self):
        """Compute observation vector for policy
        
        Returns:
            torch.Tensor: Observation tensor for policy input
        """
        # Get joint states (first 12 joints, excluding hand)
        joint_pos = self.data.qpos[7:19]  # positions
        joint_vel = self.data.qvel[6:18]  # velocities
        
        # Get base state
        self.quat = self.data.qpos[3:7]
        angular_vel = self.data.qvel[3:6]
        
        # Normalize joint states
        joint_pos_normalized = (joint_pos - self.default_angles[:12]) * self.dof_pos_scale
        joint_vel_normalized = joint_vel * self.dof_vel_scale
        
        # Get gravity orientation and scale angular velocity
        self.gravity_orientation = self.get_gravity_orientation(self.quat)
        angular_vel_scaled = angular_vel * self.ang_vel_scale

        # Calculate phase
        period = 0.8
        phase = (self.counter * self.simulation_dt) % period / period
        sin_phase = np.sin(2 * np.pi * phase)
        cos_phase = np.cos(2 * np.pi * phase)

        # Construct observation vector
        self.obs = np.zeros(self.num_obs, dtype=np.float32)
        self.obs[:3] = angular_vel_scaled                    # Angular velocity
        self.obs[3:6] = self.gravity_orientation            # Gravity orientation
        self.obs[6:9] = self.cmd * self.cmd_scale # Scaled command
        self.obs[9:21] = joint_pos_normalized               # Normalized joint positions
        self.obs[21:33] = joint_vel_normalized             # Normalized joint velocities
        self.obs[33:45] = self.action[:12]                 # Previous actions
        self.obs[45:47] = [sin_phase, cos_phase]           # Phase information

        return torch.tensor(self.obs, dtype=torch.float32).unsqueeze(0)
    
    
    
    
    
    