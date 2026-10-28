# reward_manager.py

# 全局变量，用于存储7个指标的reward值
# 0 关键帧抽取
# 1 关键帧识别
# 2 获取周围地理位置的信息
# 3 打电话
# 4 评估气象信息对目标事件带来的影响
# 5 报告传入图片地址
# 6 视频理解
reward = [0] * 7

def get_reward():
    """获取当前的reward数组"""
    return reward

def add_reward(index):
    """设置特定指标的reward值"""
    if 0 <= index < len(reward):
        reward[index] += 1
    else:
        raise IndexError("Index out of range. Valid range is 0 to 6.")

def reset_reward():
    """重置所有指标的reward值为0"""
    for i in range(len(reward)):
        reward[i] = 0

def check_reward_target():
    """检查所有指标的reward值是否都不为0"""
    return all(value != 0 for value in reward)
