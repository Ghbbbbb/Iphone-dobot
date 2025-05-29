import socket
import serial
import threading
from tkinter import Text, END
import datetime
import numpy as np
import os
import json
import time
from time import sleep
import re
import pyttsx3
import cv2
import pyrealsense2 as rs
from ultralytics import YOLO
import random
import math
from scipy.spatial.transform import Rotation as R
from transformations import quaternion_from_euler
from threading import Event

alarmControllerFile = "files/alarm_controller.json"
alarmServoFile = "files/alarm_servo.json"

# Port Feedback
MyType = np.dtype([(
    'len',
    np.int64,
), (
    'digital_input_bits',
    np.uint64,
), (
    'digital_output_bits',
    np.uint64,
), (
    'robot_mode',
    np.uint64,
), (
    'time_stamp',
    np.uint64,
), (
    'time_stamp_reserve_bit',
    np.uint64,
), (
    'test_value',
    np.uint64,
), (
    'test_value_keep_bit',
    np.float64,
), (
    'speed_scaling',
    np.float64,
), (
    'linear_momentum_norm',
    np.float64,
), (
    'v_main',
    np.float64,
), (
    'v_robot',
    np.float64,
), (
    'i_robot',
    np.float64,
), (
    'i_robot_keep_bit1',
    np.float64,
), (
    'i_robot_keep_bit2',
    np.float64,
), ('tool_accelerometer_values', np.float64, (3,)),
    ('elbow_position', np.float64, (3,)),
    ('elbow_velocity', np.float64, (3,)),
    ('q_target', np.float64, (6,)),
    ('qd_target', np.float64, (6,)),
    ('qdd_target', np.float64, (6,)),
    ('i_target', np.float64, (6,)),
    ('m_target', np.float64, (6,)),
    ('q_actual', np.float64, (6,)),
    ('qd_actual', np.float64, (6,)),
    ('i_actual', np.float64, (6,)),
    ('actual_TCP_force', np.float64, (6,)),
    ('tool_vector_actual', np.float64, (6,)),
    ('TCP_speed_actual', np.float64, (6,)),
    ('TCP_force', np.float64, (6,)),
    ('Tool_vector_target', np.float64, (6,)),
    ('TCP_speed_target', np.float64, (6,)),
    ('motor_temperatures', np.float64, (6,)),
    ('joint_modes', np.float64, (6,)),
    ('v_actual', np.float64, (6,)),
    # ('dummy', np.float64, (9, 6))])
    ('hand_type', np.byte, (4,)),
    ('user', np.byte,),
    ('tool', np.byte,),
    ('run_queued_cmd', np.byte,),
    ('pause_cmd_flag', np.byte,),
    ('velocity_ratio', np.byte,),
    ('acceleration_ratio', np.byte,),
    ('jerk_ratio', np.byte,),
    ('xyz_velocity_ratio', np.byte,),
    ('r_velocity_ratio', np.byte,),
    ('xyz_acceleration_ratio', np.byte,),
    ('r_acceleration_ratio', np.byte,),
    ('xyz_jerk_ratio', np.byte,),
    ('r_jerk_ratio', np.byte,),
    ('brake_status', np.byte,),
    ('enable_status', np.byte,),
    ('drag_status', np.byte,),
    ('running_status', np.byte,),
    ('error_status', np.byte,),
    ('jog_status', np.byte,),
    ('robot_type', np.byte,),
    ('drag_button_signal', np.byte,),
    ('enable_button_signal', np.byte,),
    ('record_button_signal', np.byte,),
    ('reappear_button_signal', np.byte,),
    ('jaw_button_signal', np.byte,),
    ('six_force_online', np.byte,),
    ('reserve2', np.byte, (82,)),
    ('m_actual', np.float64, (6,)),
    ('load', np.float64,),
    ('center_x', np.float64,),
    ('center_y', np.float64,),
    ('center_z', np.float64,),
    ('user[6]', np.float64, (6,)),
    ('tool[6]', np.float64, (6,)),
    ('trace_index', np.float64,),
    ('six_force_value', np.float64, (6,)),
    ('target_quaternion', np.float64, (4,)),
    ('actual_quaternion', np.float64, (4,)),
    ('reserve3', np.byte, (24,))])


# 读取控制器和伺服告警文件


def alarmAlarmJsonFile():
    currrntDirectory = os.path.dirname(__file__)
    jsonContrellorPath = os.path.join(currrntDirectory, alarmControllerFile)
    jsonServoPath = os.path.join(currrntDirectory, alarmServoFile)

    with open(jsonContrellorPath, encoding='utf-8') as f:
        dataController = json.load(f)
    with open(jsonServoPath, encoding='utf-8') as f:
        dataServo = json.load(f)
    return dataController, dataServo


class DobotApi:
    def __init__(self, ip, port, *args):
        self.ip = ip
        self.port = port
        self.socket_dobot = 0
        self.__globalLock = threading.Lock()
        self.text_log: Text = None
        if args:
            self.text_log = args[0]

        if self.port == 29999 or self.port == 30003 or self.port == 30004:
            try:
                self.socket_dobot = socket.socket()
                self.socket_dobot.connect((self.ip, self.port))
            except socket.error:
                print(socket.error)
                raise Exception(
                    f"Unable to set socket connection use port {self.port} !", socket.error)
        else:
            raise Exception(
                f"Connect to dashboard server need use port {self.port} !")

    def log(self, text):
        if self.text_log:
            date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S ")
            self.text_log.insert(END, date + text + "\n")
        # else:
        #     print(text)

    def send_data(self, string):
        self.log(f"Send to {self.ip}:{self.port}: {string}")
        try:
            self.socket_dobot.send(str.encode(string, 'utf-8'))
        except Exception as e:
            print(e)

    def wait_reply(self):
        """
    Read the return value
    """
        data = ""
        try:
            data = self.socket_dobot.recv(1024)
        except Exception as e:
            print(e)
        finally:
            if len(data) == 0:
                data_str = data
            else:
                data_str = str(data, encoding="utf-8")
            self.log(f'Receive from {self.ip}:{self.port}: {data_str}')
            return data_str

    def sendRecvMsg(self, string):
        """
    send-recv Sync
    """
        with self.__globalLock:
            self.send_data(string)
            recvData = self.wait_reply()
            return recvData

    def close(self):
        """
    Close the port
    """
        if (self.socket_dobot != 0):
            self.socket_dobot.close()

    def __del__(self):
        self.close()


class DobotApiDashboard(DobotApi):
    """
  Define class dobot_api_dashboard to establish a connection to Dobot
  """
    def EnableRobot(self, load=0.0, centerX=0.0, centerY=0.0, centerZ=0.0):
      string = 'EnableRobot('
      if load != 0:
        string = string + "{:f}".format(load)
        if centerX != 0 or centerY != 0 or centerZ != 0:
          string = string + ",{:f},{:f},{:f}".format(
            centerX, centerY, centerZ)
      string = string + ')'
      return self.sendRecvMsg(string)

    def DisableRobot(self):
        """
    Disabled the robot
    """
        string = "DisableRobot()"
        return self.sendRecvMsg(string)

    def ClearError(self):
        """
    Clear controller alarm information
    """
        string = "ClearError()"
        return self.sendRecvMsg(string)

    def ResetRobot(self):
        """
    Robot stop
    """
        string = "ResetRobot()"
        return self.sendRecvMsg(string)

    def SpeedFactor(self, speed):
        """
    Setting the Global rate
    speed:Rate value(Value range:1~100)
    """
        string = "SpeedFactor({:d})".format(speed)
        return self.sendRecvMsg(string)

    def User(self, index):
        """
    Select the calibrated user coordinate system
    index : Calibrated index of user coordinates
    """
        string = "User({:d})".format(index)
        return self.sendRecvMsg(string)

    def Tool(self, index):
        """
    Select the calibrated tool coordinate system
    index : Calibrated index of tool coordinates
    """
        string = "Tool({:d})".format(index)
        return self.sendRecvMsg(string)

    def RobotMode(self):
        """
    View the robot status
    """
        string = "RobotMode()"
        return self.sendRecvMsg(string)

    def PayLoad(self, weight, inertia):
        """
    Setting robot load
    weight : The load weight
    inertia: The load moment of inertia
    """
        string = "PayLoad({:f},{:f})".format(weight, inertia)
        return self.sendRecvMsg(string)

    def DO(self, index, status):
        """
    Set digital signal output (Queue instruction)
    index : Digital output index (Value range:1~24)
    status : Status of digital signal output port(0:Low level,1:High level
    """
        string = "DO({:d},{:d})".format(index, status)
        return self.sendRecvMsg(string)

    def DOExecute(self, index, status):
        """
    Set digital signal output (Instructions immediately)
    index : Digital output index (Value range:1~24)
    status : Status of digital signal output port(0:Low level,1:High level)
    """
        string = "DOExecute({:d},{:d})".format(index, status)
        return self.sendRecvMsg(string)

    def ToolDO(self, index, status):
        """
    Set terminal signal output (Queue instruction)
    index : Terminal output index (Value range:1~2)
    status : Status of digital signal output port(0:Low level,1:High level)
    """
        string = "ToolDO({:d},{:d})".format(index, status)
        return self.sendRecvMsg(string)

    def ToolDOExecute(self, index, status):
        """
    Set terminal signal output (Instructions immediately)
    index : Terminal output index (Value range:1~2)
    status : Status of digital signal output port(0:Low level,1:High level)
    """
        string = "ToolDOExecute({:d},{:d})".format(index, status)
        return self.sendRecvMsg(string)

    def AO(self, index, val):
        """
    Set analog signal output (Queue instruction)
    index : Analog output index (Value range:1~2)
    val : Voltage value (0~10)
    """
        string = "AO({:d},{:f})".format(index, val)
        return self.sendRecvMsg(string)

    def AOExecute(self, index, val):
        """
    Set analog signal output (Instructions immediately)
    index : Analog output index (Value range:1~2)
    val : Voltage value (0~10)
    """
        string = "AOExecute({:d},{:f})".format(index, val)
        return self.sendRecvMsg(string)

    def AccJ(self, speed):
        """
    Set joint acceleration ratio (Only for MovJ, MovJIO, MovJR, JointMovJ commands)
    speed : Joint acceleration ratio (Value range:1~100)
    """
        string = "AccJ({:d})".format(speed)
        return self.sendRecvMsg(string)

    def AccL(self, speed):
        """
    Set the coordinate system acceleration ratio (Only for MovL, MovLIO, MovLR, Jump, Arc, Circle commands)
    speed : Cartesian acceleration ratio (Value range:1~100)
    """
        string = "AccL({:d})".format(speed)
        return self.sendRecvMsg(string)

    def SpeedJ(self, speed):
        """
    Set joint speed ratio (Only for MovJ, MovJIO, MovJR, JointMovJ commands)
    speed : Joint velocity ratio (Value range:1~100)
    """
        string = "SpeedJ({:d})".format(speed)
        return self.sendRecvMsg(string)

    def SpeedL(self, speed):
        """
    Set the cartesian acceleration ratio (Only for MovL, MovLIO, MovLR, Jump, Arc, Circle commands)
    speed : Cartesian acceleration ratio (Value range:1~100)
    """
        string = "SpeedL({:d})".format(speed)
        return self.sendRecvMsg(string)

    def Arch(self, index):
        """
    Set the Jump gate parameter index (This index contains: start point lift height, maximum lift height, end point drop height)
    index : Parameter index (Value range:0~9)
    """
        string = "Arch({:d})".format(index)
        return self.sendRecvMsg(string)

    def CP(self, ratio):
        """
    Set smooth transition ratio
    ratio : Smooth transition ratio (Value range:1~100)
    """
        string = "CP({:d})".format(ratio)
        return self.sendRecvMsg(string)

    def LimZ(self, value):
        """
    Set the maximum lifting height of door type parameters
    value : Maximum lifting height (Highly restricted:Do not exceed the limit position of the z-axis of the manipulator)
    """
        string = "LimZ({:d})".format(value)
        return self.sendRecvMsg(string)

    def SetArmOrientation(self, r, d, n, cfg):
        """
    Set the hand command
    r : Mechanical arm direction, forward/backward (1:forward -1:backward)
    d : Mechanical arm direction, up elbow/down elbow (1:up elbow -1:down elbow)
    n : Whether the wrist of the mechanical arm is flipped (1:The wrist does not flip -1:The wrist flip)
    cfg :Sixth axis Angle identification
        (1, - 2... : Axis 6 Angle is [0,-90] is -1; [90, 180] - 2; And so on
        1, 2... : axis 6 Angle is [0,90] is 1; [90180] 2; And so on)
    """
        string = "SetArmOrientation({:d},{:d},{:d},{:d})".format(r, d, n, cfg)
        return self.sendRecvMsg(string)

    def PowerOn(self):
        """
    Powering on the robot
    Note: It takes about 10 seconds for the robot to be enabled after it is powered on.
    """
        string = "PowerOn()"
        return self.sendRecvMsg(string)

    def RunScript(self, project_name):
        """
    Run the script file
    project_name :Script file name
    """
        string = "RunScript({:s})".format(project_name)
        return self.sendRecvMsg(string)

    def StopScript(self):
        """
    Stop scripts
    """
        string = "StopScript()"
        return self.sendRecvMsg(string)

    def PauseScript(self):
        """
    Pause the script
    """
        string = "PauseScript()"
        return self.sendRecvMsg(string)

    def ContinueScript(self):
        """
    Continue running the script
    """
        string = "ContinueScript()"
        return self.sendRecvMsg(string)

    def GetHoldRegs(self, id, addr, count, type):
        """
    Read hold register
    id :Secondary device NUMBER (A maximum of five devices can be supported. The value ranges from 0 to 4
        Set to 0 when accessing the internal slave of the controller)
    addr :Hold the starting address of the register (Value range:3095~4095)
    count :Reads the specified number of types of data (Value range:1~16)
    type :The data type
        If null, the 16-bit unsigned integer (2 bytes, occupying 1 register) is read by default
        "U16" : reads 16-bit unsigned integers (2 bytes, occupying 1 register)
        "U32" : reads 32-bit unsigned integers (4 bytes, occupying 2 registers)
        "F32" : reads 32-bit single-precision floating-point number (4 bytes, occupying 2 registers)
        "F64" : reads 64-bit double precision floating point number (8 bytes, occupying 4 registers)
    """
        string = "GetHoldRegs({:d},{:d},{:d},{:s})".format(
            id, addr, count, type)
        return self.sendRecvMsg(string)

    def SetHoldRegs(self, id, addr, count, table, type=None):
        """
    Write hold register
    id :Secondary device NUMBER (A maximum of five devices can be supported. The value ranges from 0 to 4
        Set to 0 when accessing the internal slave of the controller)
    addr :Hold the starting address of the register (Value range:3095~4095)
    count :Writes the specified number of types of data (Value range:1~16)
    type :The data type
        If null, the 16-bit unsigned integer (2 bytes, occupying 1 register) is read by default
        "U16" : reads 16-bit unsigned integers (2 bytes, occupying 1 register)
        "U32" : reads 32-bit unsigned integers (4 bytes, occupying 2 registers)
        "F32" : reads 32-bit single-precision floating-point number (4 bytes, occupying 2 registers)
        "F64" : reads 64-bit double precision floating point number (8 bytes, occupying 4 registers)
    """
        if type is not None:
            string = "SetHoldRegs({:d},{:d},{:d},{:s},{:s})".format(
                id, addr, count, table, type)
        else:
            string = "SetHoldRegs({:d},{:d},{:d},{:s})".format(
                id, addr, count, table)
        return self.sendRecvMsg(string)

    def GetErrorID(self):
        """
    Get robot error code
    """
        string = "GetErrorID()"
        return self.sendRecvMsg(string)

    def DOExecute(self, offset1, offset2):
        string = "DOExecute({:d},{:d}".format(offset1, offset2) + ")"
        return self.sendRecvMsg(string)

    def ToolDO(self, offset1, offset2):
        string = "ToolDO({:d},{:d}".format(offset1, offset2) + ")"
        return self.sendRecvMsg(string)

    def ToolDOExecute(self, offset1, offset2):
        string = "ToolDOExecute({:d},{:d}".format(offset1, offset2) + ")"
        return self.sendRecvMsg(string)

    def SetArmOrientation(self, offset1):
        string = "SetArmOrientation({:d}".format(offset1) + ")"
        return self.sendRecvMsg(string)

    def SetPayload(self, offset1, *dynParams):
        string = "SetPayload({:f}".format(
            offset1)
        for params in dynParams:
            string = string + str(params) + ","
        string = string + ")"
        return self.sendRecvMsg(string)

    def PositiveSolution(self, offset1, offset2, offset3, offset4, offset5, offset6, user, tool):
        string = "PositiveSolution({:f},{:f},{:f},{:f},{:f},{:f},{:d},{:d}".format(
            offset1, offset2, offset3, offset4, offset5, offset6, user, tool) + ")"
        return self.sendRecvMsg(string)

    def InverseSolution(self, offset1, offset2, offset3, offset4, offset5, offset6, user, tool, *dynParams):
        string = "InverseSolution({:f},{:f},{:f},{:f},{:f},{:f},{:d},{:d}".format(
            offset1, offset2, offset3, offset4, offset5, offset6, user, tool)
        for params in dynParams:
            print(type(params), params)
            string = string + repr(params)
        string = string + ")"
        return self.sendRecvMsg(string)

    def SetCollisionLevel(self, offset1):
        string = "SetCollisionLevel({:d}".format(offset1) + ")"
        return self.sendRecvMsg(string)

    def GetAngle(self):
        string = "GetAngle()"
        return self.sendRecvMsg(string)

    def GetPose(self):
        string = "GetPose()"
        return self.sendRecvMsg(string)

    def EmergencyStop(self):
        string = "EmergencyStop()"
        return self.sendRecvMsg(string)

    def ModbusCreate(self, ip, port, slave_id, isRTU):
        string = "ModbusCreate({:s},{:d},{:d},{:d}".format(
            ip, port, slave_id, isRTU) + ")"
        return self.sendRecvMsg(string)

    def ModbusClose(self, offset1):
        string = "ModbusClose({:d}".format(offset1) + ")"
        return self.sendRecvMsg(string)

    def SetSafeSkin(self, offset1):
        string = "SetSafeSkin({:d}".format(offset1) + ")"
        return self.sendRecvMsg(string)

    def SetObstacleAvoid(self, offset1):
        string = "SetObstacleAvoid({:d}".format(offset1) + ")"
        return self.sendRecvMsg(string)

    def GetTraceStartPose(self, offset1):
        string = "GetTraceStartPose({:s}".format(offset1) + ")"
        return self.sendRecvMsg(string)

    def GetPathStartPose(self, offset1):
        string = "GetPathStartPose({:s}".format(offset1) + ")"
        return self.sendRecvMsg(string)

    def HandleTrajPoints(self, offset1):
        string = "HandleTrajPoints({:s}".format(offset1) + ")"
        return self.sendRecvMsg(string)

    def GetSixForceData(self):
        string = "GetSixForceData()"
        return self.sendRecvMsg(string)

    def SetCollideDrag(self, offset1):
        string = "SetCollideDrag({:d}".format(offset1) + ")"
        return self.sendRecvMsg(string)

    def SetTerminalKeys(self, offset1):
        string = "SetTerminalKeys({:d}".format(offset1) + ")"
        return self.sendRecvMsg(string)

    def SetTerminal485(self, offset1, offset2, offset3, offset4):
        string = "SetTerminal485({:d},{:d},{:s},{:d}".format(
            offset1, offset2, offset3, offset4) + ")"
        return self.sendRecvMsg(string)

    def GetTerminal485(self):
        string = "GetTerminal485()"
        return self.sendRecvMsg(string)

    def TCPSpeed(self, offset1):
        string = "TCPSpeed({:d}".format(offset1) + ")"
        return self.sendRecvMsg(string)

    def TCPSpeedEnd(self):
        string = "TCPSpeedEnd()"
        return self.sendRecvMsg(string)

    def GetInBits(self, offset1, offset2, offset3):
        string = "GetInBits({:d},{:d},{:d}".format(
            offset1, offset2, offset3) + ")"
        return self.sendRecvMsg(string)

    def GetInRegs(self, offset1, offset2, offset3, *dynParams):
        string = "GetInRegs({:d},{:d},{:d}".format(offset1, offset2, offset3)
        for params in dynParams:
            print(type(params), params)
            string = string + params[0]
        string = string + ")"
        return self.sendRecvMsg(string)

    def GetCoils(self, offset1, offset2, offset3):
        string = "GetCoils({:d},{:d},{:d}".format(
            offset1, offset2, offset3) + ")"
        return self.sendRecvMsg(string)

    def SetCoils(self, offset1, offset2, offset3, offset4):
        string = "SetCoils({:d},{:d},{:d}".format(
            offset1, offset2, offset3) + "," + repr(offset4) + ")"
        print(str(offset4))
        return self.sendRecvMsg(string)

    def DI(self, offset1):
        string = "DI({:d}".format(offset1) + ")"
        return self.sendRecvMsg(string)

    def ToolDI(self, offset1):
        string = "DI({:d}".format(offset1) + ")"
        return self.sendRecvMsg(string)

    def DOGroup(self, *dynParams):
        string = "DOGroup("
        for params in dynParams:
            string = string + str(params) + ","
        string = string + ")"
        print(string)
        return self.wait_reply()

    def BrakeControl(self, offset1, offset2):
        string = "BrakeControl({:d},{:d}".format(offset1, offset2) + ")"
        return self.sendRecvMsg(string)

    def StartDrag(self):
        string = "StartDrag()"
        return self.sendRecvMsg(string)

    def StopDrag(self):
        string = "StopDrag()"
        return self.sendRecvMsg(string)

    def LoadSwitch(self, offset1):
        string = "LoadSwitch({:d}".format(offset1) + ")"
        return self.sendRecvMsg(string)

    def wait(self, t):
        string = "wait({:d})".format(t)
        return self.sendRecvMsg(string)

    def pause(self):
        string = "pause()"
        return self.sendRecvMsg(string)

    def Continue(self):
        string = "continue()"
        return self.sendRecvMsg(string)


class DobotApiMove(DobotApi):
    """
  Define class dobot_api_move to establish a connection to Dobot
  """

    def MovJ(self, x, y, z, rx, ry, rz, *dynParams):
        """
    Joint motion interface (point-to-point motion mode)
    x: A number in the Cartesian coordinate system x
    y: A number in the Cartesian coordinate system y
    z: A number in the Cartesian coordinate system z
    rx: Position of Rx axis in Cartesian coordinate system
    ry: Position of Ry axis in Cartesian coordinate system
    rz: Position of Rz axis in Cartesian coordinate system
    """
        string = "MovJ({:f},{:f},{:f},{:f},{:f},{:f}".format(
            x, y, z, rx, ry, rz)
        for params in dynParams:
            string = string + "," + str(params)
        string = string + ")"
        print(string)
        return self.sendRecvMsg(string)

    def MovL(self, x, y, z, rx, ry, rz, *dynParams):
        """
    Coordinate system motion interface (linear motion mode)
    x: A number in the Cartesian coordinate system x
    y: A number in the Cartesian coordinate system y
    z: A number in the Cartesian coordinate system z
    rx: Position of Rx axis in Cartesian coordinate system
    ry: Position of Ry axis in Cartesian coordinate system
    rz: Position of Rz axis in Cartesian coordinate system
    """
        string = "MovL({:f},{:f},{:f},{:f},{:f},{:f}".format(
            x, y, z, rx, ry, rz)
        for params in dynParams:
            string = string + "," + str(params)
        string = string + ")"
        print(string)
        return self.sendRecvMsg(string)

    def JointMovJ(self, j1, j2, j3, j4, j5, j6, *dynParams):
        """
    Joint motion interface (linear motion mode)
    j1~j6:Point position values on each joint
    """
        string = "JointMovJ({:f},{:f},{:f},{:f},{:f},{:f}".format(
            j1, j2, j3, j4, j5, j6)
        for params in dynParams:
            string = string + "," + str(params)
        string = string + ")"
        return self.sendRecvMsg(string)

    def Jump(self):
        print("待定")

    def RelMovJ(self, offset1, offset2, offset3, offset4, offset5, offset6, *dynParams):
        """
    Offset motion interface (point-to-point motion mode)
    j1~j6:Point position values on each joint
    """
        string = "RelMovJ({:f},{:f},{:f},{:f},{:f},{:f}".format(
            offset1, offset2, offset3, offset4, offset5, offset6)
        for params in dynParams:
            string = string + "," + str(params)
        string = string + ")"
        return self.sendRecvMsg(string)

    def RelMovL(self, offsetX, offsetY, offsetZ, *dynParams):
        """
    Offset motion interface (point-to-point motion mode)
    x: Offset in the Cartesian coordinate system x
    y: offset in the Cartesian coordinate system y
    z: Offset in the Cartesian coordinate system Z
    """
        string = "RelMovL({:f},{:f},{:f}".format(offsetX, offsetY, offsetZ)
        for params in dynParams:
            string = string + "," + str(params)
        string = string + ")"
        return self.sendRecvMsg(string)

    def MovLIO(self, x, y, z, a, b, c, *dynParams):
        """
    Set the digital output port state in parallel while moving in a straight line
    x: A number in the Cartesian coordinate system x
    y: A number in the Cartesian coordinate system y
    z: A number in the Cartesian coordinate system z
    a: A number in the Cartesian coordinate system a
    b: A number in the Cartesian coordinate system b
    c: a number in the Cartesian coordinate system c
    *dynParams :Parameter Settings（Mode、Distance、Index、Status）
                Mode :Set Distance mode (0: Distance percentage; 1: distance from starting point or target point)
                Distance :Runs the specified distance（If Mode is 0, the value ranges from 0 to 100；When Mode is 1, if the value is positive,
                         it indicates the distance from the starting point. If the value of Distance is negative, it represents the Distance from the target point）
                Index :Digital output index （Value range:1~24）
                Status :Digital output state（Value range:0/1）
    """
        # example: MovLIO(0,50,0,0,0,0,(0,50,1,0),(1,1,2,1))
        string = "MovLIO({:f},{:f},{:f},{:f},{:f},{:f}".format(
            x, y, z, a, b, c)
        for params in dynParams:
            string = string + "," + str(params)
        string = string + ")"
        return self.sendRecvMsg(string)

    def MovJIO(self, x, y, z, a, b, c, *dynParams):
        """
    Set the digital output port state in parallel during point-to-point motion
    x: A number in the Cartesian coordinate system x
    y: A number in the Cartesian coordinate system y
    z: A number in the Cartesian coordinate system z
    a: A number in the Cartesian coordinate system a
    b: A number in the Cartesian coordinate system b
    c: a number in the Cartesian coordinate system c
    *dynParams :Parameter Settings（Mode、Distance、Index、Status）
                Mode :Set Distance mode (0: Distance percentage; 1: distance from starting point or target point)
                Distance :Runs the specified distance（If Mode is 0, the value ranges from 0 to 100；When Mode is 1, if the value is positive,
                         it indicates the distance from the starting point. If the value of Distance is negative, it represents the Distance from the target point）
                Index :Digital output index （Value range:1~24）
                Status :Digital output state（Value range:0/1）
    """
        # example: MovJIO(0,50,0,0,0,0,(0,50,1,0),(1,1,2,1))
        string = "MovJIO({:f},{:f},{:f},{:f},{:f},{:f}".format(
            x, y, z, a, b, c)
        self.log("Send to 192.168.5.1:29999:" + string)
        for params in dynParams:
            string = string + "," + str(params)
        string = string + ")"
        return self.sendRecvMsg(string)

    def Arc(self, x1, y1, z1, a1, b1, c1, x2, y2, z2, a2, b2, c2, *dynParams):
        """
    Circular motion instruction
    x1, y1, z1, a1, b1, c1 :Is the point value of intermediate point coordinates
    x2, y2, z2, a2, b2, c2 :Is the value of the end point coordinates
    Note: This instruction should be used together with other movement instructions
    """
        string = "Arc({:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f}".format(
            x1, y1, z1, a1, b1, c1, x2, y2, z2, a2, b2, c2)
        for params in dynParams:
            string = string + "," + str(params)
        string = string + ")"
        return self.sendRecvMsg(string)

    def Circle3(self, x1, y1, z1, a1, b1, c1, x2, y2, z2, a2, b2, c2, count, *dynParams):
        """
    Full circle motion command
    count:Run laps
    x1, y1, z1, r1 :Is the point value of intermediate point coordinates
    x2, y2, z2, r2 :Is the value of the end point coordinates
    Note: This instruction should be used together with other movement instructions
    """
        string = "Circle3({:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:f},{:d}".format(
            x1, y1, z1, a1, b1, c1, x2, y2, z2, a2, b2, c2, count)
        for params in dynParams:
            string = string + "," + str(params)
        string = string + ")"
        return self.sendRecvMsg(string)

    def ServoJ(self, j1, j2, j3, j4, j5, j6, t=0.1, lookahead_time=50, gain=500):
        """
    Dynamic follow command based on joint space
    j1~j6:Point position values on each joint

    可选参数:t、lookahead_time、gain
    t float 该点位的运行时间,默认0.1,单位:s  取值范围:[0.02,3600.0]
    lookahead_time   float 作用类似于PID的D项,默认50,标量,无单位 取值范围:[20.0,100.0]
    gain float   目标位置的比例放大器,作用类似于PID的P项,  默认500,标量,无单位   取值范围:[200.0,1000.0]
    """
        string = "ServoJ({:f},{:f},{:f},{:f},{:f},{:f},t={:f},lookahead_time={:f},gain={:f})".format(
            j1, j2, j3, j4, j5, j6, t, lookahead_time, gain)
        return self.sendRecvMsg(string)

    def ServoJS(self, j1, j2, j3, j4, j5, j6):
        """
    功能:基于关节空间的动态跟随运动。
    格式:ServoJS(J1,J2,J3,J4,J5,J6)
    """
        string = "ServoJS({:f},{:f},{:f},{:f},{:f},{:f})".format(
            j1, j2, j3, j4, j5, j6)
        return self.sendRecvMsg(string)

    def ServoP(self, x, y, z, a, b, c):
        """
    Dynamic following command based on Cartesian space
    x, y, z, a, b, c :Cartesian coordinate point value
    """
        string = "ServoP({:f},{:f},{:f},{:f},{:f},{:f})".format(
            x, y, z, a, b, c)
        return self.sendRecvMsg(string)

    def MoveJog(self, axis_id, *dynParams):
        """
    Joint motion
    axis_id: Joint motion axis, optional string value:
        J1+ J2+ J3+ J4+ J5+ J6+
        J1- J2- J3- J4- J5- J6-
        X+ Y+ Z+ Rx+ Ry+ Rz+
        X- Y- Z- Rx- Ry- Rz-
    *dynParams: Parameter Settings（coord_type, user_index, tool_index）
                coord_type: 1: User coordinate 2: tool coordinate (default value is 1)
                user_index: user index is 0 ~ 9 (default value is 0)
                tool_index: tool index is 0 ~ 9 (default value is 0)
    """
        string = "MoveJog({:s}".format(axis_id)
        for params in dynParams:
            string = string + "," + str(params)
        string = string + ")"
        return self.sendRecvMsg(string)

    def StartTrace(self, trace_name):
        """
    Trajectory fitting (track file Cartesian points)
    trace_name: track file name (including suffix)
    (The track path is stored in /dobot/userdata/project/process/trajectory/)

    It needs to be used together with `GetTraceStartPose(recv_string.json)` interface
    """
        string = f"StartTrace({trace_name})"
        return self.sendRecvMsg(string)

    def StartPath(self, trace_name, const, cart):
        """
    Track reproduction. (track file joint points)
    trace_name: track file name (including suffix)
    (The track path is stored in /dobot/userdata/project/process/trajectory/)
    const: When const = 1, it repeats at a constant speed, and the pause and dead zone in the track will be removed;
           When const = 0, reproduce according to the original speed;
    cart: When cart = 1, reproduce according to Cartesian path;
          When cart = 0, reproduce according to the joint path;

    It needs to be used together with `GetTraceStartPose(recv_string.json)` interface
    """
        string = f"StartPath({trace_name}, {const}, {cart})"
        return self.sendRecvMsg(string)

    def StartFCTrace(self, trace_name):
        """
    Trajectory fitting with force control. (track file Cartesian points)
    trace_name: track file name (including suffix)
    (The track path is stored in /dobot/userdata/project/process/trajectory/)

    It needs to be used together with `GetTraceStartPose(recv_string.json)` interface
    """
        string = f"StartFCTrace({trace_name})"
        return self.sendRecvMsg(string)

    def Sync(self):
        """
    The blocking program executes the queue instruction and returns after all the queue instructions are executed
    """
        string = "Sync()"
        return self.sendRecvMsg(string)

    def RelMovJTool(self, offset_x, offset_y, offset_z, offset_rx, offset_ry, offset_rz, tool, *dynParams):
        """
    The relative motion command is carried out along the tool coordinate system, and the end motion mode is joint motion
    offset_x: X-axis direction offset
    offset_y: Y-axis direction offset
    offset_z: Z-axis direction offset
    offset_rx: Rx axis position
    offset_ry: Ry axis position
    offset_rz: Rz axis position
    tool: Select the calibrated tool coordinate system, value range: 0 ~ 9
    *dynParams: parameter Settings（speed_j, acc_j, user）
                speed_j: Set joint speed scale, value range: 1 ~ 100
                acc_j: Set acceleration scale value, value range: 1 ~ 100
                user: Set user coordinate system index
    """
        string = "RelMovJTool({:f},{:f},{:f},{:f},{:f},{:f}, {:d}".format(
            offset_x, offset_y, offset_z, offset_rx, offset_ry, offset_rz, tool)
        for params in dynParams:
            print(type(params), params)
            string = string + ", SpeedJ={:d}, AccJ={:d}, User={:d}".format(
                params[0], params[1], params[2])
        string = string + ")"
        return self.sendRecvMsg(string)

    def RelMovLTool(self, offset_x, offset_y, offset_z, offset_rx, offset_ry, offset_rz, tool, *dynParams):
        """
    Carry out relative motion command along the tool coordinate system, and the end motion mode is linear motion
    offset_x: X-axis direction offset
    offset_y: Y-axis direction offset
    offset_z: Z-axis direction offset
    offset_rx: Rx axis position
    offset_ry: Ry axis position
    offset_rz: Rz axis position
    tool: Select the calibrated tool coordinate system, value range: 0 ~ 9
    *dynParams: parameter Settings（speed_l, acc_l, user）
                speed_l: Set Cartesian speed scale, value range: 1 ~ 100
                acc_l: Set acceleration scale value, value range: 1 ~ 100
                user: Set user coordinate system index
    """
        string = "RelMovLTool({:f},{:f},{:f},{:f},{:f},{:f}, {:d}".format(
            offset_x, offset_y, offset_z, offset_rx, offset_ry, offset_rz, tool)
        for params in dynParams:
            print(type(params), params)
            string = string + ", SpeedJ={:d}, AccJ={:d}, User={:d}".format(
                params[0], params[1], params[2])
        string = string + ")"
        return self.sendRecvMsg(string)

    def RelMovJUser(self, offset_x, offset_y, offset_z, offset_rx, offset_ry, offset_rz, user, *dynParams):
        """
    The relative motion command is carried out along the user coordinate system, and the end motion mode is joint motion
    offset_x: X-axis direction offset
    offset_y: Y-axis direction offset
    offset_z: Z-axis direction offset
    offset_rx: Rx axis position
    offset_ry: Ry axis position
    offset_rz: Rz axis position

    user: Select the calibrated user coordinate system, value range: 0 ~ 9
    *dynParams: parameter Settings（speed_j, acc_j, tool）
                speed_j: Set joint speed scale, value range: 1 ~ 100
                acc_j: Set acceleration scale value, value range: 1 ~ 100
                tool: Set tool coordinate system index
    """
        string = "RelMovJUser({:f},{:f},{:f},{:f},{:f},{:f}, {:d}".format(
            offset_x, offset_y, offset_z, offset_rx, offset_ry, offset_rz, user)
        for params in dynParams:
            string = string + "," + str(params)
        string = string + ")"
        return self.sendRecvMsg(string)

    def RelMovLUser(self, offset_x, offset_y, offset_z, offset_rx, offset_ry, offset_rz, user, *dynParams):
        """
    The relative motion command is carried out along the user coordinate system, and the end motion mode is linear motion
    offset_x: X-axis direction offset
    offset_y: Y-axis direction offset
    offset_z: Z-axis direction offset
    offset_rx: Rx axis position
    offset_ry: Ry axis position
    offset_rz: Rz axis position
    user: Select the calibrated user coordinate system, value range: 0 ~ 9
    *dynParams: parameter Settings（speed_l, acc_l, tool）
                speed_l: Set Cartesian speed scale, value range: 1 ~ 100
                acc_l: Set acceleration scale value, value range: 1 ~ 100
                tool: Set tool coordinate system index
    """
        string = "RelMovLUser({:f},{:f},{:f},{:f},{:f},{:f}, {:d}".format(
            offset_x, offset_y, offset_z, offset_rx, offset_ry, offset_rz, user)
        for params in dynParams:
            string = string + "," + str(params)
        string = string + ")"
        return self.sendRecvMsg(string)

    def RelJointMovJ(self, offset1, offset2, offset3, offset4, offset5, offset6, *dynParams):
        """
    The relative motion command is carried out along the joint coordinate system of each axis, and the end motion mode is joint motion
    Offset motion interface (point-to-point motion mode)
    j1~j6:Point position values on each joint
    *dynParams: parameter Settings（speed_j, acc_j, user）
                speed_j: Set Cartesian speed scale, value range: 1 ~ 100
                acc_j: Set acceleration scale value, value range: 1 ~ 100
    """
        string = "RelJointMovJ({:f},{:f},{:f},{:f},{:f},{:f}".format(
            offset1, offset2, offset3, offset4, offset5, offset6)
        for params in dynParams:
            string = string + "," + str(params)
        string = string + ")"
        return self.sendRecvMsg(string)


# Feedback interface
# 反馈数据接口类


class DobotApiFeedBack(DobotApi):
    def __init__(self, ip, port, *args):
        super().__init__(ip, port, *args)
        self.__MyType = []
        self.last_recv_time = time.perf_counter()
        

    def feedBackData(self):
        """
        返回机械臂状态
        Return the robot status
        """
        self.socket_dobot.setblocking(True)  # 设置为阻塞模式
        data = bytes()
        #current_recv_time = time.perf_counter() #计时，获取当前时间
        temp = self.socket_dobot.recv(144000) #缓冲区
        if len(temp) > 1440:    
            temp = self.socket_dobot.recv(144000)
        #print("get:",len(temp))
        i=0
        if len(temp) < 1440:
            while i < 5 :
                #print("重新接收")
                temp = self.socket_dobot.recv(144000)
                if len(temp) > 1440:
                    break
                i+=1
            if i >= 5:
                raise Exception("接收数据包缺失，请检查网络环境")
        
        #interval = (current_recv_time - self.last_recv_time) * 1000  # 转换为毫秒
        #self.last_recv_time = current_recv_time
        #print(f"Time interval since last receive: {interval:.3f} ms")
        
        data = temp[0:1440] #截取1440字节
        #print(len(data))
        #print(f"Single element size of MyType: {MyType.itemsize} bytes")
        self.__MyType = None   

        if len(data) == 1440:        
            self.__MyType = np.frombuffer(data, dtype=MyType)

        return self.__MyType
        


class MyRobot:
    def __init__(self, ip="192.168.5.1", dashboard_port=29999, move_port=30003, feed_port=30004):
        self.ip = ip
        self.dashboard_port = dashboard_port
        self.move_port = move_port
        self.feed_port = feed_port
        self.dashboard = None
        self.move = None
        self.feed = None
        self.obj_pose_data = None
        self.obj_json_file = "obj_poses.json"
        self.obj_txt_file = "./prompts/dobot3/scene.txt"
        self.ser = serial.Serial(
            port='COM3',
            baudrate=115200,
            bytesize=serial.EIGHTBITS,
            stopbits=serial.STOPBITS_ONE,
            timeout=1
        )
        self.model = YOLO("yolo11x.pt")

        '''
        realsense连接配置
        '''
        # 创建一个RealSense pipeline
        self.pipeline = rs.pipeline()
        # 配置管道
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        # 启动管道
        self.profile = self.pipeline.start(self.config)
        # 创建一个align对象，用于将深度流和颜色流对齐
        self.align = rs.align(rs.stream.color)
        print("RealSense initialized successfully.")

        self.Hcb = np.array([
            [0.7114498378116707, 0.7026840236116416, -0.008619236549793999, -0.06442096586241067],
            [-0.7022274970572221, 0.7113475752226492, 0.02934569136358208, -0.02720603316209926],
            [0.02675202150299365, -0.01482532245218585, 0.99953216014278, -0.1461301182996755],
            [0, 0, 0, 1]])

        self.threshold = 10  # 位置变化阈值（单位：毫米）
        self.step_points = 5  # 插值点数
        self.current_target = None  # 当前追踪目标的缓存
        self.grab_interrupt = Event()  # 新增中断事件
        self.grab_timeout = 1.0        # 抓取超时时间
        self.stop_flag = False  # 在__init__中初始化


        
    def connect(self):
        """建立连接并返回dashboard, move, feed对象"""
        try:
            print("正在建立连接...")
            self.dashboard = DobotApiDashboard(self.ip, self.dashboard_port)
            self.move = DobotApiMove(self.ip, self.move_port)
            self.feed = DobotApi(self.ip, self.feed_port)
            print(">.<连接成功>!<")
            print("开始使能...")
            self.dashboard.EnableRobot()
            print('清空队列')
            self.dashboard.ResetRobot()
            print("回到起始点")
            self.reset_robot()
            ##获得realsense数据
            print("初始化模型预测...")
            self.take_photo()

            self.load_obj_pose_from_file()
            print("加载位姿文件")
            return self.dashboard, self.move, self.feed
        except Exception as e:
            print(f"连接失败: {str(e)}")
            raise e
    
    def get_target_coordinate(self):
        # while True:
        frames = self.pipeline.wait_for_frames()  # 等待获取图像帧
        aligned_frames = self.align.process(frames)  # 获取对齐帧
        depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的depth帧
        color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的color帧

        # 相机参数的获取
        intrinsic = color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参
        depth_intrinsic = depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）
        camera_parameters = {'fx': intrinsic.fx, 'fy': intrinsic.fy, 'ppx': intrinsic.ppx, 'ppy': intrinsic.ppy,
                                'height': intrinsic.height, 'width': intrinsic.width,
                                'depth_scale': self.profile.get_device().first_depth_sensor().get_depth_scale()
                                }
        # 保存内参到本地
        with open('./intrinsics.json', 'w') as fp:
            json.dump(camera_parameters, fp)

        # 识别部分
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        results = self.model.predict(color_image,conf=0.15)
        results[0].show()
        boxes = results[0].boxes.xyxy
        classes = results[0].boxes.cls 

        result_list = []

        for box, clss in zip(boxes, classes):
            name = self.model.names[int(clss)]
            if name=="sports ball":
                name ="yellow_block"
            ##画
            cv2.rectangle(color_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            dist = self.get_mid_pos(color_image, box, depth_image, 24)
            # 获取目标物的距离相机的距离
            distance = name + str(dist / 1000)[:4] + 'm'
            # 如果绘制的文字超出了画面最上边的显示的范围，进行调整
            if box[1] < 10:
                cv2.putText(color_image, distance, (int(box[0]), int(box[1]) + 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2)
            else:
                cv2.putText(color_image, distance, (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2)
            
            #计算
            x1, y1, x2, y2, x3, y3 =box[0], box[1], box[2], box[3], (box[0] + box[2]) / 2, (box[1] + box[3]) / 2  # 计算中心点
            x, y = int(x3), int(y3)
            dis = depth_frame.get_distance(x, y)  # 获取深度值
            camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrinsic, [x, y], dis)


            result_list.append(f"{name},{', '.join(map(str, camera_coordinate))}")  # 按格式存入列表
            # result_list.append(f"{name},{x},{y},{dis}")  # 按格式存入列表

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        # 拼接RGB图和深度图
        images = np.hstack((color_image, depth_colormap))
        cv2.imshow("Detection Result", images)
        cv2.waitKey(5000)  # 显示 1000ms（1秒）
        cv2.destroyAllWindows()  # 关闭窗口
        print(result_list)
        return ",".join(result_list)  # 连接成字符串返回


    def get_valid_depth(self, depth_frame, x, y, search_radius=5):
        """在指定点周围搜索有效深度"""
        valid_depths = []
        for dx in range(-search_radius, search_radius+1):
            for dy in range(-search_radius, search_radius+1):
                if 0 <= x+dx < 640 and 0 <= y+dy < 480:  # 假设分辨率640x480
                    d = depth_frame.get_distance(x+dx, y+dy)
                    if 0.3 < d < 3.0:  # 有效距离范围
                        valid_depths.append(d)
        if valid_depths:
            return np.mean(valid_depths)
        else:
            return None


    def get_mid_pos(self,frame, box, depth_data, randnum):
        distance_list = []
        mid_pos = [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2]
        min_val = min(abs(box[2] - box[0]), abs(box[3] - box[1]))
        for i in range(randnum):
            bias = random.randint(-min_val // 4, min_val // 4)
            dist = depth_data[int(mid_pos[1] + bias), int(mid_pos[0] + bias)]
            cv2.circle(frame, (int(mid_pos[0] + bias), int(mid_pos[1] + bias)), 4, (255, 0, 0), -1)
            if dist:
                distance_list.append(dist)
        distance_list = np.array(distance_list)
        distance_list = np.sort(distance_list)[randnum // 2 - randnum // 4:randnum // 2 + randnum // 4]
        return np.mean(distance_list)

    def pose_to_homogeneous_matrix(self, position, quaternion):
        """
        将机械臂末端位姿（位置+四元数）转换为齐次变换矩阵
        :param position: [x, y, z] 单位：米
        :param quaternion: [x, y, z, w] 四元数（注意顺序！）
        :return: 4x4齐次变换矩阵 (基座坐标系←机械臂末端坐标系)
        """
        # 创建旋转对象（注意四元数顺序为x,y,z,w）
        rotation = R.from_quat(quaternion)
        # 生成3x3旋转矩阵
        rot_matrix = rotation.as_matrix()
        # 构建齐次矩阵
        homogeneous_matrix = np.eye(4)
        homogeneous_matrix[:3, :3] = rot_matrix
        homogeneous_matrix[:3, 3] = position
        return homogeneous_matrix
    
    def transform_point_to_base(self, H_base_gripper, Hcb, point_cam):
        """
        将相机坐标系下的点转换到机械臂基座坐标系
        :param H_base_gripper: 机械臂末端到基座的齐次矩阵 (4x4)
        :param Hcb: 相机到机械臂末端的齐次矩阵 (4x4)
        :param point_cam: 相机坐标系下的点 [x, y, z] 单位：米
        :return: 基座坐标系下的坐标 [x, y, z]
        """
        # 将点转换为齐次坐标
        # print("相机:",point_cam)
        # print("机械臂:",H_base_gripper)

        point_cam_homogeneous = np.append(point_cam, 1.0)
        # 坐标变换：P_base = H_base_gripper * Hcb * P_cam
        point_base_homogeneous = H_base_gripper @ Hcb @ point_cam_homogeneous
        # 转换为3D坐标
        return point_base_homogeneous[:3]

    def update_json_with_realsense_data(self, data_str):
        """
        将相机数据字符串解析并写入 JSON 文件

        :param data_str: 字符串数据，如 "烧杯,98.820808,-468.939209,试管,-4.843419,-477.638275"
        """
       # 获取原始机械臂位姿（单位：mm和度）
        x_mm, y_mm, z_mm, euler_w, euler_p, euler_r = self.get_current_pose()[:6]

        # 转换为米单位的位置
        gripper_position = [
            x_mm / 1000,
            y_mm / 1000,
            z_mm / 1000
        ]

        # 将欧拉角转换为四元数（注意顺序和单位转换）
        quaternion = quaternion_from_euler(
            math.radians(euler_w),    # 绕X轴旋转（假设是roll）
            math.radians(euler_p),    # 绕Y轴旋转（假设是pitch）
            math.radians(euler_r),    # 绕Z轴旋转（假设是yaw）
            axes='sxyz'              # 旋转顺序为X-Y-Z（根据实际机械臂配置调整）
        )

        # 调整四元数顺序为 [qx, qy, qz, qw]
        gripper_quaternion = [
            quaternion[1],  # qx
            quaternion[2],  # qy
            quaternion[3],  # qz
            quaternion[0]   # qw
        ]
        H_base_gripper = self.pose_to_homogeneous_matrix(gripper_position, gripper_quaternion)
        # 将字符串解析为列表
        data_list = data_str.split(",")

        # 将数据转换为 JSON 格式
        json_data = {}

        # 遍历数据并按名称填充 JSON
        for i in range(0, len(data_list), 4):
            name = data_list[i]
            x = float(data_list[i + 1])
            y = float(data_list[i + 2])
            z = float(data_list[i + 3])
            object_in_cam = [x,y,z]
            object_in_base = self.transform_point_to_base(H_base_gripper, self.Hcb, object_in_cam)


            json_data[name] = [object_in_base[0]*1000, object_in_base[1]*1000, object_in_base[2]*1000, euler_w, euler_p, euler_r]

        # 将数据写入 JSON 文件
        try:
            # 如果 JSON 文件存在，加载原数据进行合并
            with open(self.obj_json_file, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_data = {}

        # 合并新数据
        existing_data.update(json_data)

        # 将合并后的数据写回文件
        with open(self.obj_json_file, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=4)

        print(f"数据已成功写入到 {self.obj_json_file}")
        self.load_obj_pose_from_file()
        self.update_text_from_json(self.obj_json_file, self.obj_txt_file)

    def take_photo(self):
        data_str = self.get_target_coordinate()
        # 按逗号分割整个字符串
        split_data = data_str.split(',')
        # 每4个字段为一个物体（名称,x,y,z），提取名称
        names = [
            split_data[i] 
            for i in range(0, len(split_data) - 3, 4)  # 步长为4，确保不越界
        ]
        self.update_json_with_realsense_data(data_str)  # 保持原始参数格式
        return names

     
    def reset_robot(self):
        """重置机器人，停止当前运动并回到 Home 点"""
        # 返回到 Home 点
        # self.move.JointMovJ(11.6743, 18.2755, -52.5398, -50.6371, 88.4353, 54.7007)
        Home = [188,-109,310.5329,179.6885,0.6219,-132.8713]
        self.run_point(Home)

    def run_point(self, point_list):
        """让机器人移动到指定点"""
        self.move.MovJ(point_list[0], point_list[1], point_list[2],
                        point_list[3], point_list[4], point_list[5])
        self.wait_for_arrival(point_list)
        

    def wait_for_arrival(self, target_point, timeout=20):
        """等待机器人到达指定位置"""
        start_time = time.time()
        while True:
            # 获取当前坐标
            # print("time:",time.time() - start_time)
            current_pose = self.get_current_pose()
            # 比较当前位置和目标位置
            if self._check_arrival(current_pose, target_point):
                print("目标位置到达!")
                return
            if time.time() - start_time > timeout:
                print("超时未到达目标位置!")
                print(f"当前坐标: {current_pose}, 目标坐标: {target_point}")
                return
            sleep(0.1)
    
    
    def get_pose_coordinates(self,pose_str):
        # 使用正则表达式从字符串中提取坐标部分
        match = re.search(r"\{([^}]+)\}", pose_str)
        if match:
            # 提取出的坐标部分是一个字符串，按照逗号分割并转换为浮点数
            coordinates = list(map(float, match.group(1).split(',')))
            return coordinates
        return None
    
    def _check_arrival(self, current_pose, target_point, tolerance=1):
        """检查机器人是否到达目标位置，基于误差总和"""
        error = sum(abs(current_pose[i] - target_point[i]) for i in range(3))
        print("error:",error)
        return error <= tolerance
    
    def gripper_ctrl(self, cmd: str):
        if cmd == "打开":
            command = bytes.fromhex("7b010200204920012c1c7d")
        elif cmd == "关闭":
            command = bytes.fromhex("7b010201204920012c1d7d")
        self.ser.write(command)
        self.ser.flush()
        sleep(0.1)

        
    def get_current_pose(self):
        return self.get_pose_coordinates(self.dashboard.GetPose())
    
    def get_current_joint(self):
        return self.get_pose_coordinates(self.dashboard.GetAngle())
    
    def load_obj_pose_from_file(self):
        """从文件加载物体的位姿"""
        with open(self.obj_json_file, "r", encoding="utf-8") as f:
            self.obj_pose_data = json.load(f)

    def get_obj_pose(self, obj_name):
        """根据物体名称返回物体的位姿"""
        return self.obj_pose_data.get(obj_name, None)  # 若找不到返回None

    def grab(self, obj_name):
        obj_pose = self.get_obj_pose(obj_name)
        self.run_point(obj_pose)
        self.dashboard.SpeedL(10)
        self.move.RelMovLTool(0,0,30,0,0,0,9)
        self.move.Sync()
        self.gripper_ctrl("关闭")
        sleep(1)
        self.run_point(obj_pose)

    def grab2(self, obj_name):
            # 生物实验
            # obj_pose = [327.2448,-184.4612,162.7304,180,0,-90]
        obj_pose = self.get_obj_pose(obj_name)
        self.run_point(np.add(obj_pose, np.array([0, 0, 120, 0, 0, 0])))
        self.run_point(obj_pose)
        self.gripper_ctrl("关闭")
        sleep(1)
        self.run_point(np.add(obj_pose, np.array([0, 0, 120, 0, 0, 0])))
    
    def say(self,conversation):
        self.text_to_speech(conversation.replace('_',''))
        # print(conversation)

    ###############skill_update###########################
    def move_gripper_orientation(self,direction,distance):
        """
        This function moves the gripper to the corresponding position based on direction and distance. The direction is chosen from 'forward', 'backward', 'left', 'right', 'upward', 'downward', and the distance is measured in centimeters.
        """
        current_pose = self.get_current_pose()
        
        direction_map = {
            '前': np.array([1, 0, 0 ,0 ,0 ,0]),
            '后': np.array([-1, 0, 0, 0, 0, 0]),
            '左': np.array([0, 1, 0, 0, 0, 0]),
            '右': np.array([0, -1, 0, 0, 0, 0]),
            '上': np.array([0, 0, 1, 0, 0, 0]),
            '下': np.array([0, 0, -1, 0, 0, 0])
        }
        
        if direction in direction_map:
            move_vector = direction_map[direction] * distance
        else:
            raise ValueError("Invalid direction")
        
        target_pos = current_pose + move_vector
        self.run_point(target_pos)

    def stack_object_on_object(self, top_object_name, base_object_name):
        """
        This function stack top_object on base_object. It grabs the top_object, moves it above the base_object, and 
    then places it on the base_object.
        """
        # Grab the top block
        self.grab(top_object_name)

        # Get the position and quaternion of the base block
        base_pose = self.get_obj_pose(base_object_name)

        # Calculate the new position for placing the top block on the base block
        new_pose_block = base_pose + np.array([0, 0, 80+38.7, 0, 0, 0])
        self.run_point(new_pose_block)
        new_pose_block2 = base_pose + np.array([0, 0, 40+38.7, 0, 0, 0])

        # new_pose_block2 = base_pose + np.array([0, 0, 5+38.7, 0, 0, 0])
        self.run_point(new_pose_block2)
        self.gripper_ctrl("打开")
        # self.update_json_object_position(top_object_name)
        self.run_point(new_pose_block)
        self.reset_robot()


    def move_joint_by_increment(self, joint_name, angle_increment):
        """
        让指定的关节按照用户输入的角度进行增量旋转。

        参数:
            joint_name (str): 关节名称，如 "1轴", "2轴", ..., "6轴"
            angle_increment (float): 旋转的角度，正数表示正转，负数表示反转。
        """
        # 读取当前机械臂各关节角度
        current_angles_str = self.dashboard.GetAngle()
        
        # 解析返回的数据，提取关节角度
        try:
            # 获取角度字符串部分 "{16.754065,54.203350,-133.353714,-10.707003,89.956627,16.847198}"
            angles_str = current_angles_str.split("{")[1].split("}")[0]
            # 转换为浮点数列表
            current_angles = [float(angle) for angle in angles_str.split(",")]
        except (IndexError, ValueError):
            print("错误: 无法解析当前关节角度数据")
            return

        # 关节名称映射到索引
        joint_mapping = {"1轴": 0, "2轴": 1, "3轴": 2, "4轴": 3, "5轴": 4, "6轴": 5}

        # 确保输入的关节名称有效
        if joint_name not in joint_mapping:
            print(f"错误: 关节名称 {joint_name} 无效，应为 1轴~6轴")
            return

        # 获取对应的关节索引
        joint_index = joint_mapping[joint_name]

        # 计算新的角度
        new_angle = current_angles[joint_index] + angle_increment

        # 更新关节角度列表
        current_angles[joint_index] = new_angle

        # 让机械臂执行新的角度指令
        self.move.JointMovJ(*current_angles)
        self.move.Sync()


    def pick_and_place_next_to(self, obj_to_pick, obj_reference, direction, distance):
        """
        Picks up an object and places it next to another object in the specified direction and distance.        

        Parameters:
            obj_to_pick (str): Name of the object to pick.
            obj_reference (str): Name of the reference object next to which the picked object will be placed.   
            direction (str): Direction to place the picked object relative to the reference object. Options: 'left', 'right', 'forward', 'backward'.
            distance (float): Distance in centermeters from the reference object.
        """
        self.grab(obj_to_pick)
        reference_pose = self.get_obj_pose(obj_reference)

        direction_map = {
            '前': np.array([1, 0, 0 ,0 ,0 ,0]),
            '后': np.array([-1, 0, 0, 0, 0, 0]),
            '左': np.array([0, 1, 0, 0, 0, 0]),
            '右': np.array([0, -1, 0, 0, 0, 0]),
        }

        if direction in direction_map:
            displacement = direction_map[direction] * distance
        else:
            raise ValueError("Invalid direction")

        place_position = reference_pose + displacement + np.array([0, 0, 70, 0, 0, 0])
        self.run_point(place_position)
        place_position2 = reference_pose + displacement + np.array([0, 0, 20, 0, 0, 0])
        self.run_point(place_position2)
        self.gripper_ctrl("打开")
        # self.update_json_object_position(obj_to_pick)
        self.run_point(place_position)
        self.reset_robot()


    def pour(self, pick_obj, place_obj):
        # Grab the top block
        point1 = np.array([274.3589, -30.9175, 289.1548, -90, 45, -73])
        point2 = np.array([155.659,-303.5431,311.1830,-88.5,44.24,-90.447])
        # point3 = np.array([155.659, -132.76, 348, -88, 44, -90])
        self.run_point(point2)

        self.grab2(pick_obj)

        pick_pose = self.get_obj_pose(pick_obj)
        new_pose = pick_pose + np.array([0, 0, 250, 0, 0, 0])
        self.run_point(new_pose)
        self.run_point(point1)
        self.dashboard.SpeedFactor(25)
        self.move_joint_by_increment('6轴',-120)
        print("执行完 -110° 旋转")
        self.dashboard.SpeedFactor(100)
        for i in range(3):
            self.move_joint_by_increment('6轴',0.25)
            self.move_joint_by_increment('6轴',-0.25)
        self.run_point(point1)
        self.dashboard.SpeedFactor(15)
        self.run_point(new_pose)
        self.run_point(pick_pose+np.array([0, 0, 10, 0, 0, 0]))
        self.gripper_ctrl("打开")
        sleep(0.5)
        self.run_point(pick_pose+np.array([0, 0, 100, 0, 0, 0]))
        self.run_point(point2)


    def update_json_object_position(self, top_object_name):
        """更新 JSON 文件中 top_object_name 的位置，使其与 base_object_name 的 x, y 对齐，并增加 z 值"""
        
        with open(self.obj_json_file, "r", encoding="utf-8") as f:
            obj_data = json.load(f)
        # 获取 base_object 的 x, y, z
        base_x, base_y, base_z, base_rx, base_ry, base_rz = self.get_current_pose()
        
        # 计算新的 z 值
        new_z = base_z - 5
        
        # 更新 top_object 的位姿
        obj_data[top_object_name] = [base_x, base_y, new_z, base_rx, base_ry, base_rz]
        
        # 写回 JSON 文件
        with open(self.obj_json_file, "w", encoding="utf-8") as f:
            json.dump(obj_data, f, ensure_ascii=False, indent=4)
        
        print(f"已更新 {top_object_name} 位置: x={base_x}, y={base_y}, z={new_z}")
        self.load_obj_pose_from_file()

    def update_text_from_json(self, json_path, txt_path):
        """
        读取JSON文件中的物体名称，并填充到txt文本模板中。

        参数：
            json_path (str): JSON文件的路径
            txt_template (str): 带有占位符的文本模板

        返回：
            str: 填充后的文本
        """
        # 读取JSON文件
        with open(json_path, "r", encoding="utf-8") as f:
            obj_data = json.load(f)

        # 获取物体名称列表
        object_names = list(obj_data.keys())
        print(object_names)

        # 计算物体数量
        x = len(object_names)

        # 生成填充的字符串
        object_str = '", "'.join(object_names)

    # 文本模板
        updated_text = f'''#场景:抓取与放置
    你现在在一个房间里，房间里有一个末端装有双指夹爪和3d摄像头的机械臂和一个工作台，上面随机放置了{x}个不同的物体，
    分别叫做"{object_str}"。 现在你需要听从我的指令，控制机械臂完成任务。'''

        # 写入到txt文件，覆盖原文件
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(updated_text)
        print("覆盖完毕")
    

    def text_to_speech(self,text):
        # 初始化语音引擎
        engine = pyttsx3.init()
        
        # 设置语速
        rate = engine.getProperty('rate')
        engine.setProperty('rate', rate)
        
        # 设置音量
        volume = engine.getProperty('volume')
        engine.setProperty('volume', volume + 0.25)
        
        # 播放文本
        engine.say(text)
        engine.runAndWait()


    def euclidean_distance(self, p1, p2):
        """计算三维欧氏距离"""
        return math.sqrt(sum((a - b)**2 for a, b in zip(p1, p2)))

    def linear_interpolate(self, start_xyz, end_xyz, euler_angles):
        """生成带欧拉角的线性插值路径"""
        x = np.linspace(start_xyz[0], end_xyz[0], self.step_points)
        y = np.linspace(start_xyz[1], end_xyz[1], self.step_points)
        z = np.linspace(start_xyz[2], end_xyz[2], self.step_points)
        return [[xi, yi, zi, *euler_angles] for xi, yi, zi in zip(x, y, z)]
