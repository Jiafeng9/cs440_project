import time
import mujoco
import numpy as np
import torch
import yaml
import argparse
import mujoco.viewer

class G1WalkingShower:
    def __init__(self, config_file):
        """Initialize the robot controller with configuration file"""
        # Load configuration
        with open(f"{config_file}.yaml", "r", encoding="utf-8") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
            
        # Model paths
        self.policy_path = self.config["policy_path"]
        self.xml_path = self.config["xml_path"]
        
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
        self.mj_model = None
        self.mj_data = None
        self.viewer = None
        self.cmd = None
        self.obs = None
        self.action = None
        self.counter = 0
        self.quat = None
        self.target_dof_pos = None
        self.gravity_orientation = None
        self.policy = None
        self.initial_position = None

    def get_gravity_orientation(self, quaternion):
        """Calculate gravity orientation from quaternion"""
        qw, qx, qy, qz = quaternion
        
        gravity_orientation = np.zeros(3)
        gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
        gravity_orientation[1] = -2 * (qz * qy + qw * qx)
        gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
        
        return gravity_orientation
                                                      # 0
    def pd_control(self, target_pos, current_pos, kp, target_vel, current_vel, kd):
        """Calculate torques using PD control
            The PD controller is responsible for smoothly moving the 
            joints to these continuously updated target positions.
            
            Like:Please move to this new location, but move smoothly and stop when you reach it
        
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
        joint_pos = self.mj_data.qpos[7:19]  # positions
        joint_vel = self.mj_data.qvel[6:18]  # velocities
        
        # Get base state
        self.quat = self.mj_data.qpos[3:7]
        angular_vel = self.mj_data.qvel[3:6]
        
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

    def run_simulation(self):
        """Run the main simulation loop"""
        # Initialize simulation state
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.target_dof_pos = self.default_angles.copy()
        self.obs = np.zeros(self.num_obs, dtype=np.float32)
        
        # Load robot model and policy
        self.mj_model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = self.simulation_dt
        self.policy = torch.jit.load(self.policy_path)
        self.initial_position = self.mj_data.qpos[:2]  # initial x_y position
        
        print("\n=== Joint name and current angle ===")
        joint_qpos = self.mj_data.qpos #[7:]  # 从第7个开始是关节角度
        print("joint_qpos.size",joint_qpos.size)
        
        # Get target position from user
        cmd_input = input("Enter target position with three values separated by space: ")
        self.cmd = np.array(cmd_input.split(), dtype=np.float32)
        
        # 打印维度信息以进行调试
        print(f"Model joints: {self.mj_model.njnt}")
        print(f"Default angles length: {len(self.default_angles)}")
        print(f"KP length: {len(self.kps)}")
        print(f"KD length: {len(self.kds)}")
        
        with mujoco.viewer.launch_passive(self.mj_model, self.mj_data) as viewer:
            start_time = time.time()
            last_print_time = time.time()
            
            # 记录初始位置和计算目标
            initial_position = self.mj_data.qpos[:2]
            target_position = self.cmd[:2] + initial_position
            target_yaw = self.cmd[2]  # 目标旋转角度
            

            
            while viewer.is_running() and time.time() - start_time < self.simulation_duration:
                step_start = time.time()
                
                # 获取当前状态
                current_position = self.mj_data.qpos[:2]
                current_quat = self.mj_data.qpos[3:7]
                current_yaw = 2 * np.arctan2(current_quat[3], current_quat[0])  # 从四元数计算yaw角
                
                # 计算位置和角度误差
                position_error = np.linalg.norm(target_position - current_position)
                yaw_error = abs(target_yaw - current_yaw)
                
                # 每秒打印一次状态
                if time.time() - last_print_time > 1.0:
                    print(f"\nCurrent Position:")
                    print(f"Position: {current_position}, Error: {position_error:.3f}")
                    print(f"Rotation angle: {current_yaw:.2f}, Error: {yaw_error:.3f}")
                    last_print_time = time.time()

                if (position_error < 0.1 and yaw_error < 0.3):  # 10cm阈值
                    print("\n=== Reach the goal! ===")
                    print(f"Final position: {current_position}")
                    print(f"Final angle: {current_yaw:.2f}")
                    print(f"Position error: {position_error:.3f}")
                    print(f"Angle error: {yaw_error:.3f}")
                    break
                    
                # 计算控制力矩
                tau = self.pd_control(
                    self.target_dof_pos,
                    self.mj_data.qpos[7:37],  # not excluding hand
                    self.kps,
                    np.zeros(30),            # target velocity is zero
                    self.mj_data.qvel[6:36], # not excluding hand
                    self.kds
                )
                self.mj_data.ctrl[:] = tau
                
                # 应用控制力矩
                # Step physics
                mujoco.mj_step(self.mj_model, self.mj_data)
                
                self.counter += 1
                if self.counter % self.control_decimation == 0:
                    # Update policy
                    obs_tensor = self.compute_observation()
                    #self.action = self.policy(obs_tensor).detach().numpy().squeeze()
                    action_12dof = self.policy(obs_tensor).detach().numpy().squeeze()
                    # 扩展动作到43自由度
                    self.action = np.zeros(30, dtype=np.float32)
                    self.action[:12] = action_12dof  # 只使用前12个关节的动作
                    self.action[12:] = 0  # 手部关节保持为0
                    self.target_dof_pos = self.action * self.action_scale + self.default_angles
                
                # Update viewer
                viewer.sync()
                
                # Time keeping
                time_until_next_step = self.mj_model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    def main(self):
        """Main control loop"""
        while True:
            self.run_simulation()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot walking controller")
    parser.add_argument("config_file", type=str, help="Configuration file name (without .yaml extension)")
    args = parser.parse_args()
    
    controller = G1WalkingShower(args.config_file)
    controller.main()