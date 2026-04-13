#!/usr/bin/env python3
"""
GPU Utilization Monitor
实时监控指定GPU的利用率，并计算平均值

Usage:
    # 监控单个GPU
    python monitor_gpu_utilization.py --gpu-id 3 --output logs/gpu_util.log
    
    # 监控时指定采样间隔
    python monitor_gpu_utilization.py --gpu-id 3 --interval 0.1 --output logs/gpu_util.log
"""

import argparse
import time
import signal
import sys
from datetime import datetime
from typing import Optional

try:
    import pynvml
except ImportError:
    print("Error: pynvml not installed. Install with: pip install nvidia-ml-py3")
    sys.exit(1)


class GPUMonitor:
    def __init__(self, gpu_id: int, interval: float = 0.1, output_file: Optional[str] = None):
        """
        初始化GPU监控器
        
        Args:
            gpu_id: GPU设备ID
            interval: 采样间隔(秒)
            output_file: 输出日志文件路径
        """
        self.gpu_id = gpu_id
        self.interval = interval
        self.output_file = output_file
        self.running = True
        self.samples = []
        
        # 初始化NVML
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        
        # 获取GPU信息
        self.gpu_name = pynvml.nvmlDeviceGetName(self.handle)
        if isinstance(self.gpu_name, bytes):
            self.gpu_name = self.gpu_name.decode('utf-8')
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """处理中断信号"""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Received signal {signum}, stopping monitor...")
        self.running = False
    
    def get_utilization(self) -> dict:
        """获取当前GPU利用率信息"""
        util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
        memory = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        temperature = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
        power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # mW to W
        
        return {
            'timestamp': time.time(),
            'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            'gpu_util': util.gpu,  # GPU利用率 (%)
            'mem_util': util.memory,  # 显存带宽利用率 (%)
            'mem_used_gb': memory.used / 1024**3,  # 已使用显存 (GB)
            'mem_total_gb': memory.total / 1024**3,  # 总显存 (GB)
            'temperature': temperature,  # 温度 (°C)
            'power': power  # 功耗 (W)
        }
    
    def monitor(self):
        """开始监控"""
        print(f"="*80)
        print(f"GPU Monitor Started")
        print(f"GPU ID: {self.gpu_id} ({self.gpu_name})")
        print(f"Sampling Interval: {self.interval}s")
        print(f"Output File: {self.output_file if self.output_file else 'None (stdout only)'}")
        print(f"="*80)
        print(f"{'Time':<20} {'GPU%':<8} {'Mem%':<8} {'MemUsed(GB)':<12} {'Temp(°C)':<10} {'Power(W)':<10}")
        print(f"-"*80)
        
        # 打开输出文件
        log_file = None
        if self.output_file:
            try:
                log_file = open(self.output_file, 'w')
                log_file.write(f"timestamp,datetime,gpu_util,mem_util,mem_used_gb,mem_total_gb,temperature,power\n")
                log_file.flush()
            except Exception as e:
                print(f"Warning: Cannot open output file {self.output_file}: {e}")
                log_file = None
        
        try:
            while self.running:
                data = self.get_utilization()
                self.samples.append(data)
                
                # 打印到终端
                print(f"{data['datetime']:<20} {data['gpu_util']:<8} {data['mem_util']:<8} "
                      f"{data['mem_used_gb']:<12.2f} {data['temperature']:<10} {data['power']:<10.1f}")
                
                # 写入文件
                if log_file:
                    log_file.write(f"{data['timestamp']},{data['datetime']},{data['gpu_util']},"
                                 f"{data['mem_util']},{data['mem_used_gb']:.2f},{data['mem_total_gb']:.2f},"
                                 f"{data['temperature']},{data['power']:.1f}\n")
                    log_file.flush()
                
                time.sleep(self.interval)
                
        except Exception as e:
            print(f"\nError during monitoring: {e}")
        finally:
            if log_file:
                log_file.close()
            self._print_statistics()
            pynvml.nvmlShutdown()
    
    def _print_statistics(self):
        """打印统计信息"""
        if not self.samples:
            print("\nNo samples collected.")
            return
        
        gpu_utils = [s['gpu_util'] for s in self.samples]
        mem_utils = [s['mem_util'] for s in self.samples]
        mem_useds = [s['mem_used_gb'] for s in self.samples]
        temps = [s['temperature'] for s in self.samples]
        powers = [s['power'] for s in self.samples]
        
        duration = self.samples[-1]['timestamp'] - self.samples[0]['timestamp']
        
        print(f"\n{'='*80}")
        print(f"Monitoring Statistics (GPU {self.gpu_id})")
        print(f"{'='*80}")
        print(f"Total Duration:       {duration:.2f}s")
        print(f"Total Samples:        {len(self.samples)}")
        print(f"Sampling Rate:        {len(self.samples) / duration:.2f} samples/s")
        print(f"-"*80)
        print(f"GPU Utilization:")
        print(f"  Average:            {sum(gpu_utils) / len(gpu_utils):.2f}%")
        print(f"  Min:                {min(gpu_utils):.2f}%")
        print(f"  Max:                {max(gpu_utils):.2f}%")
        print(f"-"*80)
        print(f"Memory Utilization:")
        print(f"  Average:            {sum(mem_utils) / len(mem_utils):.2f}%")
        print(f"  Min:                {min(mem_utils):.2f}%")
        print(f"  Max:                {max(mem_utils):.2f}%")
        print(f"-"*80)
        print(f"Memory Usage (GB):")
        print(f"  Average:            {sum(mem_useds) / len(mem_useds):.2f} GB")
        print(f"  Min:                {min(mem_useds):.2f} GB")
        print(f"  Max:                {max(mem_useds):.2f} GB")
        print(f"-"*80)
        print(f"Temperature (°C):")
        print(f"  Average:            {sum(temps) / len(temps):.2f}°C")
        print(f"  Min:                {min(temps):.2f}°C")
        print(f"  Max:                {max(temps):.2f}°C")
        print(f"-"*80)
        print(f"Power (W):")
        print(f"  Average:            {sum(powers) / len(powers):.2f}W")
        print(f"  Min:                {min(powers):.2f}W")
        print(f"  Max:                {max(powers):.2f}W")
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Monitor GPU utilization in real-time")
    parser.add_argument('--gpu-id', type=int, default=0,
                       help='GPU device ID to monitor (default: 0)')
    parser.add_argument('--interval', type=float, default=0.1,
                       help='Sampling interval in seconds (default: 0.1)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output log file path (optional)')
    
    args = parser.parse_args()
    
    monitor = GPUMonitor(gpu_id=args.gpu_id, interval=args.interval, output_file=args.output)
    monitor.monitor()


if __name__ == '__main__':
    main()
