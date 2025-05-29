from dobot_api import MyRobot,DobotApiFeedBack
import numpy as np
from time import sleep


# 示例代码：
if __name__ == '__main__':
    robot = MyRobot()

    try:
        dashboard, move, feed = robot.connect()

        #test1.语音交互
        robot.say("你好，我是你的人工智能助手")

        # #test2.拍摄图像
        # robot.take_photo()

        # #test3.夹爪开闭
        # robot.gripper_ctrl("关闭")
        # sleep(5)
        # robot.gripper_ctrl("打开")

        # #test4.简单运动-世界坐标系
        # robot.move_gripper_orientation("foward", 50)   #前
        # robot.move_gripper_orientation("left", 10)   #左

        # #test5.简单运动-关节坐标系
        # robot.move_joint_by_increment("1轴",10) #J1轴正转10°
        # robot.move_joint_by_increment("2轴",-10) #J2轴负转10°

        # #test6.复杂运动(在机械臂视野范围内放入香蕉(banana)和鼠标(mouse))
        # robot.take_photo()
        # robot.stack_object_on_object("banana","mouse")  
        # robot.reset_robot()  

        
    except Exception as e:
        if str(e) == "unsupported operand type(s) for +: 'NoneType' and 'int'":
            print("机器人操作发生错误: 环境中未找到该物体！")
        else:
            print(f"机器人操作发生错误: {str(e)}")
