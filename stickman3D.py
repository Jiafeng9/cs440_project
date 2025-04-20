import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation as R


class stickman3D:
    def __init__(self, npz_file, npz_motion_file, only_joint=True, whole_body=False,
                 joint_regressor="J_regressor", v_template="v_template", kintree_table="kintree_table"):
        '''
        Initialize 3D skelton model and load motion data.
        Parameters:
            npz_file: SMPLX model data file path
            npz_motion_file: action data file path (npz format)
            only_joint: if only show joints
            whole_body: if show whole body 
            joint_regressor: A sparse matrix used to "linearly combine" the positions of 55 joints from 10,475 vertices  (55,10475)
            v_template: Template mesh vertex positions (the position of each point) (10475, 3)
            kintree_table: kintree table name (2,55)
        '''
        self.npz_file = npz_file
        self.npz_motion_file = npz_motion_file
        self.only_joint = only_joint
        self.whole_body = False if self.only_joint else whole_body
        self.joint_regressor = joint_regressor
        self.v_template = v_template
        self.kintree_table = kintree_table        
        self.model_data = None
        self.motion_data = None
        self.initial_positions = None
        self.colors = {}
        self.kintree_table_data = None
        self.tree_graph = None
        
        # load data 
        self._load_data()
        self._load_default_positions()
        # create sketlon tree
        self.tree_graph = self.tree(self.kintree_table_data)
        # set colors
        self._set_colors()
    
    
    
    
    def process_frame(self, frame_idx):
        """
        Deal with one frame of motion data and get joint positions.
        
        Parameter:
            frame_idx: int    
        Return:
            numpy.ndarray: the joint positions after processing
        """
        if self.motion_data is None:
            raise Exception("No motion data loaded")
        
        if 'poses' not in self.motion_data:
            raise Exception("No pose data in motion data")
        
        poses = self.motion_data['poses']
        
        if frame_idx >= poses.shape[0]:
           raise ValueError(f"frame index{frame_idx}out of range. Total frames: {poses.shape[0]}")
        # get the pose of the current frame (2485,165)
        frame_pose = poses[frame_idx]
        # apply the pose to the skeleton
        transformed_positions = self.apply_pose_to_skeleton(frame_pose)
        return transformed_positions
    
    
    def apply_pose_to_skeleton(self, pose):
        from scipy.spatial.transform import Rotation as R
        import numpy as np
        
        # 初始化变换后的位置
        transformed_positions = np.copy(self.initial_positions)
        
        # 应用全局旋转
        # 提取姿态数据中的全局旋转（前3个值）
        # pose格式是[global_orient(3), body_pose(NUM_BODY_JOINTS*3), ...]
        global_rot = R.from_rotvec(pose[:3])
        for i in range(len(transformed_positions)):
            transformed_positions[i] = global_rot.apply(transformed_positions[i])
        
        # 创建变换字典
        transforms = {}
        transforms[0] = np.eye(4)  # 根关节变换矩阵
        
        # 直接使用预先计算的关节父子关系
        joint_parents = {}
        for i in range(self.kintree_table_data.shape[1]):
            child = self.kintree_table_data[1, i]
            parent = self.kintree_table_data[0, i]
            joint_parents[child] = parent
        
        # 按层级处理关节（从根到叶）
        level_joints = [0]  # 第0层只有根关节
        processed = {0}
        
        # 处理所有关节
        while len(processed) < min(55, len(pose) // 3):
            next_level = []
            for joint in level_joints:
                # 找到所有以当前关节为父关节的子关节
                for child, parent in joint_parents.items():
                    if parent == joint and child not in processed:
                        next_level.append(child)
                        
                        # 处理子关节
                        if 3*child+3 <= len(pose):
                            # 获取局部旋转
                            joint_rot = R.from_rotvec(pose[3*child:3*child+3])
                            rot_matrix = joint_rot.as_matrix()
                            
                            # 计算并应用变换
                            rel_pos = self.initial_positions[child] - self.initial_positions[parent]
                            if parent > 0:
                                parent_global_rot = R.from_matrix(transforms[parent][:3, :3])
                                rel_pos = parent_global_rot.apply(rel_pos)
                            
                            rotated_rel_pos = joint_rot.apply(rel_pos)
                            transformed_positions[child] = transformed_positions[parent] + rotated_rel_pos
                            
                            # 更新变换矩阵
                            transform = np.eye(4)
                            transform[:3, :3] = rot_matrix
                            transform[:3, 3] = transformed_positions[child] - transformed_positions[parent]
                            
                            if parent > 0:
                                transforms[child] = np.dot(transforms[parent], transform)
                            else:
                                transforms[child] = transform
                            
                            processed.add(child)
            
            level_joints = next_level
        
        return transformed_positions
    

    

    def animate_skeleton_2d(self, start_frame=0, end_frame=None, fps=30, save_path=None, width_size=11, height_size=11):
        """
        从动作数据中创建2D骨架动画
        
        参数:
            start_frame: 起始帧索引
            end_frame: 结束帧索引，如果为None则使用所有可用帧
            fps: 每秒帧数
            save_path: 如果提供，保存动画到该路径
            width_size: 图像宽度
            height_size: 图像高度
            
        返回:
            matplotlib.animation.Animation: 动画对象
        """
        if self.motion_data is None:
            raise Exception("No motion data loaded")
        
        if 'poses' not in self.motion_data:
            raise Exception("No pose data in motion data")
        
        poses = self.motion_data['poses']
        total_frames = poses.shape[0]  # 2484 
        
        if end_frame is None:
            end_frame = total_frames
        
        if start_frame < 0 or start_frame >= total_frames:
            raise ValueError(f"start frame{start_frame} out of range。Total frames: {total_frames}")
        
        if end_frame <= start_frame or end_frame > total_frames:
            raise ValueError(f"End frame{end_frame} out of range。Must > {start_frame} and <= {total_frames}")
        
        frame_count = end_frame - start_frame
        

        fig = plt.figure(figsize=(width_size, height_size))
        ax = fig.add_subplot(111)
        
        # 通过采样帧计算动态轴限制
        sample_frames = np.linspace(start_frame, end_frame-1, min(10, frame_count), dtype=int)
        all_positions = []
        for frame_idx in sample_frames:
            all_positions.append(self.process_frame(frame_idx))
        all_positions = np.vstack(all_positions)
        
        min_x, min_y = np.min(all_positions[:, :2], axis=0) - 0.1
        max_x, max_y = np.max(all_positions[:, :2], axis=0) + 0.1
        
        # 初始化绘图元素
        joint_dots = ax.scatter([], [], c="blue", alpha=0.5)
        lines = []
        
        # 为骨架中的每个连接创建线条
        for key, paths in self.tree_graph.joint_list.items():
            for path in paths:
                # 连接关节与路径的第一个节点
                if len(path) > 0:
                    line, = ax.plot([], [], color=self.colors[key], linewidth=2)
                    lines.append((key, [path[0]], line))
                
                # 连接路径中的相邻节点
                for i in range(len(path)-1):
                    line, = ax.plot([], [], color=self.colors[key], linewidth=2)
                    lines.append((key, [path[i], path[i+1]], line))
        
        # 设置轴限制
        ax.set_xlim([min_x, max_x])
        ax.set_ylim([min_y, max_y])
        
        # 带帧计数器的标题
        title = ax.set_title(f'帧: 0/{frame_count}')
        ax.set_axis_off()
        
        def init():
            """初始化动画"""
            joint_dots.set_offsets(np.zeros((0, 2)))
            for _, _, line in lines:
                line.set_data([], [])
            title.set_text(f'帧: 0/{frame_count}')
            return [joint_dots] + [line for _, _, line in lines] + [title]
        
        def update(frame):
            """更新每帧的动画"""
            # 处理当前帧以获取关节位置
            frame_idx = start_frame + frame
            joint_positions = self.process_frame(frame_idx)
            
            # 更新散点图(2D)
            joint_dots.set_offsets(joint_positions[:, :2])
            
            # 更新线条
            for key, path_indices, line in lines:
                if len(path_indices) == 1:
                    # 关节和路径第一个节点之间的线
                    x_data = [joint_positions[key][0], joint_positions[path_indices[0]][0]]
                    y_data = [joint_positions[key][1], joint_positions[path_indices[0]][1]]
                else:
                    # 路径中相邻节点之间的线
                    x_data = [joint_positions[path_indices[0]][0], joint_positions[path_indices[1]][0]]
                    y_data = [joint_positions[path_indices[0]][1], joint_positions[path_indices[1]][1]]
                
                line.set_data(x_data, y_data)
            
            # 更新标题
            title.set_text(f'帧: {frame+1}/{frame_count} ({frame_idx})')
            
            return [joint_dots] + [line for _, _, line in lines] + [title]
        
        # 创建动画
        anim = animation.FuncAnimation(
            fig, update, frames=frame_count, init_func=init, 
            interval=1000/fps, blit=True
        )
        
        # 根据请求保存动画
        if save_path:
            self.save_animation(anim, save_path, fps)
        
        plt.tight_layout()
        plt.show()
        
        return anim

    def save_animation(self, anim, save_path, fps=30):
        """
        保存动画到文件
        
        参数:
            anim: 要保存的动画
            save_path: 保存动画的路径
            fps: 每秒帧数
        """
        try:
            writer = 'ffmpeg' if save_path.endswith('.mp4') else 'pillow'
            anim.save(save_path, fps=fps, writer=writer)
            print(f"动画已保存到 {save_path}")
        except Exception as e:
            print(f"保存动画失败: {e}")
            print("确保已安装所需依赖:")
            print("MP4格式需要: ffmpeg (pip install ffmpeg-python)")
            print("GIF格式需要: pillow (pip install pillow)")
            raise
        
    def animate_skeleton_3d(self, start_frame=0, end_frame=None, fps=30, save_path=None, width_size=11, height_size=11):
        """
        从动作数据中创建3D骨架动画
        
        参数:
            start_frame: 起始帧索引
            end_frame: 结束帧索引，如果为None则使用所有可用帧
            fps: 每秒帧数
            save_path: 如果提供，保存动画到该路径
            width_size: 图像宽度
            height_size: 图像高度
            
        返回:
            matplotlib.animation.Animation: 动画对象
        """
        if self.motion_data is None:
            raise Exception("未加载动作数据")
        
        if 'poses' not in self.motion_data:
            raise Exception("动作数据中未找到姿态数据")
        
        poses = self.motion_data['poses']
        total_frames = poses.shape[0]  # 新格式为 (2484, 165)
        
        if end_frame is None:
            end_frame = total_frames
        
        if start_frame < 0 or start_frame >= total_frames:
            raise ValueError(f"起始帧{start_frame}超出范围。总帧数: {total_frames}")
        
        if end_frame <= start_frame or end_frame > total_frames:
            raise ValueError(f"结束帧{end_frame}超出范围。必须 > {start_frame} 且 <= {total_frames}")
        
        frame_count = end_frame - start_frame
        
        # 创建动画图形
        fig = plt.figure(figsize=(width_size, height_size))
        ax = fig.add_subplot(111, projection='3d')
        
        # 通过采样帧计算动态轴限制
        sample_frames = np.linspace(start_frame, end_frame-1, min(10, frame_count), dtype=int)
        all_positions = []
        for frame_idx in sample_frames:
            all_positions.append(self.process_frame(frame_idx))
        all_positions = np.vstack(all_positions)
        
        min_vals = np.min(all_positions, axis=0) - 0.1
        max_vals = np.max(all_positions, axis=0) + 0.1
        
        # 初始化绘图元素
        joint_dots = ax.scatter([], [], [], c="blue", alpha=0.5)
        lines = []
        
        # 为骨架中的每个连接创建线条
        for key, paths in self.tree_graph.joint_list.items():
            for path in paths:
                # 连接关节与路径的第一个节点
                if len(path) > 0:
                    line, = ax.plot([], [], [], color=self.colors[key], linewidth=2)
                    lines.append((key, [path[0]], line))
                
                # 连接路径中的相邻节点
                for i in range(len(path)-1):
                    line, = ax.plot([], [], [], color=self.colors[key], linewidth=2)
                    lines.append((key, [path[i], path[i+1]], line))
        
        # 设置轴限制
        ax.set_xlim([min_vals[0], max_vals[0]])
        ax.set_ylim([min_vals[1], max_vals[1]])
        ax.set_zlim([min_vals[2], max_vals[2]])
        
        # 带帧计数器的标题
        title = ax.set_title(f'帧: 0/{frame_count}')
        
        def init():
            """初始化动画"""
            joint_dots._offsets3d = ([], [], [])
            for _, _, line in lines:
                line.set_data([], [])
                line.set_3d_properties([])
            title.set_text(f'帧: 0/{frame_count}')
            return [joint_dots] + [line for _, _, line in lines] + [title]
        
        def update(frame):
            """更新每帧的动画"""
            # 处理当前帧以获取关节位置
            frame_idx = start_frame + frame
            joint_positions = self.process_frame(frame_idx)
            
            # 更新散点图
            joint_dots._offsets3d = (joint_positions[:, 0], joint_positions[:, 1], joint_positions[:, 2])
            
            # 更新线条
            for key, path_indices, line in lines:
                if len(path_indices) == 1:
                    # 关节和路径第一个节点之间的线
                    x_data = [joint_positions[key][0], joint_positions[path_indices[0]][0]]
                    y_data = [joint_positions[key][1], joint_positions[path_indices[0]][1]]
                    z_data = [joint_positions[key][2], joint_positions[path_indices[0]][2]]
                else:
                    # 路径中相邻节点之间的线
                    x_data = [joint_positions[path_indices[0]][0], joint_positions[path_indices[1]][0]]
                    y_data = [joint_positions[path_indices[0]][1], joint_positions[path_indices[1]][1]]
                    z_data = [joint_positions[path_indices[0]][2], joint_positions[path_indices[1]][2]]
                
                line.set_data(x_data, y_data)
                line.set_3d_properties(z_data)
            
            # 更新标题
            title.set_text(f'帧: {frame+1}/{frame_count} ({frame_idx})')
            
            return [joint_dots] + [line for _, _, line in lines] + [title]
        
        # 创建动画
        anim = animation.FuncAnimation(
            fig, update, frames=frame_count, init_func=init, 
            interval=1000/fps, blit=True
        )
        
        # 根据请求保存动画
        if save_path:
            self.save_animation(anim, save_path, fps)
        
        plt.tight_layout()
        plt.show()
        
        return anim
    
        
    ######################################
    #  Private Methods
    ######################################
    def _load_data(self):
        try:
            if not os.path.exists(self.npz_file):
                raise FileNotFoundError("Could not find model data file")
            if not os.path.exists(self.npz_motion_file):
                raise FileNotFoundError("Could not find motion data file")
            
            self.model_data = np.load(self.npz_file)
            self.motion_data = np.load(self.npz_motion_file)
            
        except FileNotFoundError as e:
            print(f"Loaded data failed: {e}")
            raise
        except ValueError as e:
            print(f"Data format error: {e}")
            raise
        except Exception as e:
            print(f"Unknown error: {e}")
            raise
    
    def _load_default_positions(self):
        """
            Load the default positions of the joints and adjust the coordinate system.
        """
        if self.joint_regressor not in self.model_data:
            raise Exception(f"Joinr regressor {self.joint_regressor} not found in model data")
        
        if self.v_template not in self.model_data:
            raise Exception(f"Vertex template {self.v_template} not found in model data")
        
        if self.kintree_table not in self.model_data:
            raise Exception(f"Kintree table {self.kintree_table} not found in model data")
            
        self.kintree_table_data = self.model_data[self.kintree_table]
        self.initial_positions = self.model_data[self.joint_regressor].dot(self.model_data[self.v_template])
        
        # # 坐标系统调整 - SMPLX通常使用Y轴向上，可能需要调整坐标
        # # 调整高度比例 (Y轴)
        # height_scale = 0.85  # 这个值可能需要调整
        # self.initial_positions[:, 1] *= height_scale
        
        # # 检查基本身体关节是否在合理范围内 (可选)
        # # 膝盖关节的位置应该在身高的约一半处
        # # 检查大约的身高 (从头到脚的距离)
        # # head_idx = 15  # 根据SMPLX模型，这可能是头部关节索引
        # # foot_idx = 10  # 根据SMPLX模型，这可能是脚部关节索引
        # # total_height = np.abs(self.initial_positions[head_idx, 1] - self.initial_positions[foot_idx, 1])
        # # print(f"调整后的身体高度: {total_height}")


    def _set_colors(self):
        """
            Set the color of the joints 
        """
        for key in self.tree_graph.joint_list.keys():
            self.colors[key] = np.random.rand(3)  # (R,G,B)
    
    
    
    
    
    #########################################
    #  Helper Classes
    #########################################
    class tree(object):
        class _Node(object):  
            def __init__(self, item):
                self.item = item
                self.children = []
                self.parent = None

            def add_child(self, child):
                child.parent = self
                self.children.append(child)
            
        def __init__(self, relation_2d_matrix, initial_point=0):
            """
            Assume the initial point is the first one in the matrix
            
            Parameters:
                relation_2d_matrix: relation 2d matrix
                initial_point: initial point(int)
            """
            self.initial_point = initial_point
            self.root = self._Node(self.initial_point)
            self.relation_2d_matrix = relation_2d_matrix
            self.nodes = {self.initial_point: self.root}    # {item:node}
            self.joint_list= {}             #     {joint_1: [[child_id1, child_id2,...]
                                            #                  [child_id3, child_id4,...]]    
                                            #      
                                            #      joint_2: [[]]       
                                            #                                               }                
            
            self.build_tree()
            self.create_joint_list()
            self.create_whole_list()
            
        def add_node(self, node):
            if node.item not in self.nodes:
                self.nodes[node.item] = node

        def create_joint_list(self):
            for key in self.nodes:
                if len(self.nodes[key].children) > 1:
                    self.joint_list[key] = []


        def create_whole_list(self):
            """
                Recursively generate the whole path for trajectory
            """
            
            def traverse_path(node, current_path):
                """
                    Base case 1 : if reachs to new joint node(len path >1)
                    Base case 2 : if reachs to leaf node(node.childeren is empty)
                """
                # init, including start node   
                path = current_path + [node.item]
                
                # reach joint node (path is not only the joint >1)
                if node.item in self.joint_list and len(current_path) > 1:
                    # old joint node
                    parent_id = current_path[0] 
                    self.joint_list[parent_id].append(current_path[1:])
                    return 
                        
                # leaf node
                elif not node.children and current_path:
                    parent_id = current_path[0]
                    # add leaf node  
                    self.joint_list[parent_id].append(current_path[1:] + [node.item])
                    
                else:
                    for child in node.children:
                        traverse_path(child, path)
            
            for joint_id in self.joint_list:
                traverse_path(self.nodes[joint_id], [])

        def build_tree(self):
            for i in range(len(self.relation_2d_matrix[0])):
                parent_id = self.relation_2d_matrix[0][i]
                child_id = self.relation_2d_matrix[1][i]

                # skip the initial point 
                if child_id == self.initial_point:
                    continue

                # if parent exists,create child 
                if parent_id in self.nodes:
                    parent_node = self.nodes[parent_id]
                    child_node = self._Node(child_id)
                    parent_node.add_child(child_node)  
                    self.add_node(child_node)
                    
        def print_tree(self):
            def print_node(node, level=0):
                if level == 0:
                    print(str(node.item))  
                else:
                    print('  ' * (level-1) + '|-' + str(node.item))  
                
                for child in node.children:
                    print_node(child, level+1)
            
            if self.root:
                print_node(self.root)
            else:
                print("Tree is empty")


if __name__=="__main__":
    model_file_path = "/Users/jiafeng/Desktop/cs440_robotic_project/outside_project/PantoMatrix/datasets/smplx/SMPLX_NEUTRAL_2020.npz"
    motion_file_path = "/Users/jiafeng/Desktop/cs440_robotic_project/outside_project/PantoMatrix/examples/motion/output_output.npz"
    
    # 创建stickman3D实例
    stick = stickman3D(model_file_path, motion_file_path)
    # stick.tree_graph.print_tree()

    # # 生成3D动画（第0-100帧，30fps）
    # stick.animate_skeleton_3d(start_frame=0, end_frame=100, fps=30)

    # # 生成2D动画并保存为MP4文件
    # stick.animate_skeleton_2d(save_path="animation.mp4")