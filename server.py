import errno
import os
import sys
import re
import logging
import socket
import colorama
colorama.init(autoreset=True)
logging.basicConfig(level=logging.INFO)
sys.path.append(os.path.dirname(__file__))

from dobot_api import MyRobot


def extract_python_code(content):
    """ Extract the python code from the input content.
    :param content: message contains the code from gpt's reply.
    :return(str): python code if the content is correct
    """
    code_block_regex = re.compile(r"```(.*?)```", re.DOTALL)
    code_blocks = code_block_regex.findall(content)
    if code_blocks:
        full_code = "\n".join(code_blocks)
        if full_code.startswith("python"):
            full_code = full_code[7:]
        return full_code
    else:
        return None


def execute_python_code(pri, code):
    """ Execute python code with the input content.
    :param pri(Class): class name in prompts.
    :param code(str): python method to call.
    """
    print("\n"'\033[34m'"Please wait while I run the code in Sim...")
    print("\033[34m""code:" + code)
    try:
        exec(code)
        print('\033[32m'"Done!\n")
        return True
    except Exception as e:
        logging.warning('\033[31m'"Found error while running the code: {}".format(e))
        return False


def main():
    logging.info("Initializing TCP...")
    HOST = ''
    ss = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ss.bind((HOST, 5001))
    ss.listen(1)
    ss.setblocking(0)
    logging.info("Done.")

    logging.info("Initializing Robot...")
    robot = MyRobot()
    robot.connect()
    logging.info("Done.")

    while True:
        try:
            logging.info("Waiting for connection...")
            conn, addr = ss.accept()
            conn.setblocking(0)
            logging.info(f"Connected by {addr}")

            while True:
                try:
                    data = conn.recv(2048)
                    if not data:
                        break

                    code = extract_python_code(data.decode())

                    if code is not None:
                        success = execute_python_code(robot, code)  # 执行提取到的代码

                        if not success:
                            robot.reset_robot()

                    else:
                        logging.warning('\033[31m'":No code extracted or user interrupt.")

                    conn.sendall("ACK".encode())  # 发送ACK确认消息

                except socket.error as e:
                    if e.errno == errno.EWOULDBLOCK:
                        pass
                    else:
                        raise
        except socket.error as e:
            if e.errno == errno.EWOULDBLOCK:
                pass
            else:
                raise


if __name__ == "__main__":
    main()