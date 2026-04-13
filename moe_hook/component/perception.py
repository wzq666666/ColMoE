import psutil
import pynvml
import concurrent.futures
import os
import subprocess
import json
from typing import Optional
import time

class ResourcePerception:
    def __init__(self):
        pynvml.nvmlInit()
        self.device_count = pynvml.nvmlDeviceGetCount()
        self.thread_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.thread_executor.submit(self.memory_guard)
    
    # 感知存储信息，返回值（总大小，已使用，剩余） 
    def get_memory_info(self):
        mem = psutil.virtual_memory()
        return [mem.total, mem.used, mem.available]
    
    def get_device_memory_info(self):
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return [mem_info.total, mem_info.used, mem_info.free]
        except pynvml.NVMLError:
            return None
        
    def memory_guard(self, mem_thresh=300):
        while True:
            rest_mem = self.get_memory_info()[2] / (1024**2)
            rest_device_mem = self.get_device_memory_info()[2] / (1024**2)
            # print(f"rest_mem: {rest_mem:.2f} MB, rest_device_mem: {rest_device_mem:.2f} MB")
            if rest_device_mem < mem_thresh:
                print(f"[MemoryGuard] ⚠️    Low GPU memory: {rest_device_mem:.2f} MB. exit!!!!")
                os._exit(1)  # 强制退出推理
            if rest_mem < mem_thresh:
                print(f"[MemoryGuard] ⚠️    Low CPU memory: {rest_mem:.2f} MB. exit!!!!")
                os._exit(1)  # 强制退出推理
            time.sleep(0.5)
            
    # 感知计算能力信息，返回值（原始计算能力FLOPS，当前利用率）
    def get_compute_info(self):
        flops = None

        # 2. 当前利用率
        cpu_util = psutil.cpu_percent(interval=None) # 非阻塞式获取瞬时值
        
        gpu_utils = []
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_utils.append({'device_id': 0, 'gpu_util': util.gpu})
        except pynvml.NVMLError:
            gpu_utils.append({'device_id': 0, 'gpu_util': None})
        
        utilization = {'cpu': cpu_util, 'gpu': gpu_utils}
        
        return [flops, utilization]
    # 感知带宽信息是，返回值（cpu-gpu之间的带宽，cpu-gpu之间的时延，网络带宽，网络时延）
    def __passive_calculate_bandwidth(self,interval_seconds=0.5):
        """
        基于网络IO计数器，被动计算并打印在指定时间间隔内的网络带宽。
        若没有网络流量的收发，则无法正确感知当前带宽
        """
        # 第一次读取
        last_io = psutil.net_io_counters()
        last_time = time.time()

        # 等待指定的时间
        time.sleep(interval_seconds)

        # 第二次读取
        current_io = psutil.net_io_counters()
        current_time = time.time()

        # 计算时间差和字节差
        elapsed_time = current_time - last_time
        bytes_sent_diff = current_io.bytes_sent - last_io.bytes_sent
        bytes_recv_diff = current_io.bytes_recv - last_io.bytes_recv

        # 避免除以零
        if elapsed_time == 0:
            print("时间间隔为零，无法计算带宽。")
            return

        # 计算速度 (单位: 字节/秒)
        send_speed_bps = bytes_sent_diff / elapsed_time
        recv_speed_bps = bytes_recv_diff / elapsed_time

        # 转换为更易读的单位，例如 Mbps (兆比特每秒)
        # 1 byte = 8 bits,  1 Megabit = 1,000,000 bits
        send_speed_mbps = (send_speed_bps * 8) / 1_000_000
        recv_speed_mbps = (recv_speed_bps * 8) / 1_000_000

        print(f"时间间隔: {elapsed_time:.2f} 秒")
        print(f"发送速度: {send_speed_mbps:.2f} Mbps")
        print(f"接收速度: {recv_speed_mbps:.2f} Mbps")
        return [send_speed_mbps, recv_speed_mbps]

    def __active_calculate_bandwidth(server_ip: str, port: int = 5201) -> Optional[float]:
        """
        使用 iperf3 命令行工具测量到指定服务器的网络带宽。

        Args:
            server_ip (str): iperf3 服务器的IP地址。
            port (int): iperf3 服务器的端口。

        Returns:
            Optional[float]: 以 Mbps (Megabits per second) 为单位的平均带宽。
                            如果测量失败或 iperf3 未安装，则返回 None。
        """
        command = [
            'iperf3',
            '-c', server_ip,
            '-p', str(port),
            '-J',  # <<< 关键参数：输出为JSON格式
            '-t', '1' # 测试时长5秒，避免过长
        ]
        
        print(f"正在执行命令: {' '.join(command)}")
        
        try:
            # 执行命令并捕获输出
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=10  # 设置一个比测试时长更长的超时时间
            )

            if result.returncode == 0:
                # 解析JSON输出
                json_output = json.loads(result.stdout)
                # 提取最后总结部分（sum_received）的比特率
                # iperf3 的单位是 bits per second
                bits_per_second = json_output['end']['sum_received']['bits_per_second']
                # 获取时延（延迟），单位为毫秒
                # iperf3 的 JSON 输出中，'end'->'streams'[0]->'receiver' 里有 'mean_rtt' 字段（单位微秒），部分版本可能没有
                try:
                    mean_rtt_us = json_output['end']['streams'][0]['receiver'].get('mean_rtt', None)
                    if mean_rtt_us is not None:
                        latency_ms = mean_rtt_us / 1000  # 转换为毫秒
                        print(f"网络时延: {latency_ms:.2f} ms")
                except Exception as e:
                    print(f"无法获取时延信息: {e}")
                mbps = bits_per_second / 1_000_000  # 转换为 Mbps
                return round(mbps, 2), round(latency_ms, 2)
            else:
                print(f"iperf3 执行出错: {result.stderr}")
                return None

        except FileNotFoundError:
            print("错误: 'iperf3' 命令未找到。请确保已经安装 iperf3。")
            return None
        except subprocess.TimeoutExpired:
            print("错误: iperf3 命令执行超时。")
            return None
        except (json.JSONDecodeError, KeyError) as e:
            print(f"错误: 解析iperf3的JSON输出失败: {e}")
            return None

    def get_communication_info(self):
        return self.__passive_calculate_bandwidth()

if __name__ == '__main__':
    # 初始化
    sensor = ResourcePerception()
    mem_total, mem_used, mem_available = sensor.get_memory_info()
    print(f"内存信息: 总计 {mem_total/1e9:.2f} GB, 已用 {mem_used/1e9:.2f} GB, 可用 {mem_available/1e9:.2f} GB")
    
    # 获取每个GPU的显存信息
    gpu_mem = sensor.get_device_memory_info()
    if gpu_mem:
        print(f"GPU 显存: 总计 {gpu_mem[0]/1e9:.2f} GB, 已用 {gpu_mem[1]/1e9:.2f} GB, 剩余 {gpu_mem[2]/1e9:.2f} GB")
    
    # 获取计算能力信息
    flops, utilization = sensor.get_compute_info()
    print(f"计算能力: ")
    print(f"  - 理论FLOPS: {flops if flops is not None else 'N/A (需基准测试)'}")
    print(f"  - CPU利用率: {utilization['cpu']}%")
    
    for gpu_util in utilization['gpu']:
        print(f"  - GPU {gpu_util['device_id']} 利用率: {gpu_util['gpu_util']}%")

    up, down = sensor.get_communication_info()
    print(f"网络带宽: 上传 {up:.2f} Mbps, 下载 {down:.2f} Mbps")