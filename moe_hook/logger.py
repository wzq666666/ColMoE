"""
Logger utilities for MOE hooks.
"""

import os
import time
from typing import Dict, Optional

_once_logged: Dict[str, bool] = {}
_log_cleared = False

# 日志级别: 0=关闭, 1=仅错误, 2=重要事件, 3=详细(每层)
# 通过环境变量 MOE_HOOK_LOG_LEVEL 控制
_LOG_LEVEL: int = int(os.environ.get('MOE_HOOK_LOG_LEVEL', '2'))

# 日志缓冲（减少文件 IO）
_log_buffer: list = []
_log_buffer_size: int = 100  # 缓冲区大小
_last_flush_time: float = 0.0
_flush_interval: float = 1.0  # 最多每秒刷新一次


def set_log_level(level: int) -> None:
    """设置日志级别: 0=关闭, 1=仅错误, 2=重要事件, 3=详细"""
    global _LOG_LEVEL
    _LOG_LEVEL = level


def get_log_level() -> int:
    """获取当前日志级别"""
    return _LOG_LEVEL


def log_once(key: str, msg: str) -> None:
    """Log a message only once per key."""
    if not _once_logged.get(key):
        print(f"[MOE-HOOK] {msg}", flush=True)
        _once_logged[key] = True


def append_log(msg: str, log_path: str = None, level: int = 3) -> None:
    """
    Append a timestamped message to the log file.
    
    Args:
        msg: 日志消息
        log_path: 日志文件路径
        level: 日志级别 (1=错误, 2=重要, 3=详细)
    """
    global _log_buffer, _last_flush_time
    
    if not log_path or level > _LOG_LEVEL:
        return
    
    try:
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        _log_buffer.append(f"[{ts}] {msg}\n")
        
        # 检查是否需要刷新缓冲区
        current_time = time.time()
        should_flush = (
            len(_log_buffer) >= _log_buffer_size or
            (current_time - _last_flush_time) >= _flush_interval
        )
        
        if should_flush:
            _flush_log_buffer(log_path)
            _last_flush_time = current_time
            
    except Exception as e:
        log_once('logfile_err', f"Failed to write log file {log_path}: {e}")


def _flush_log_buffer(log_path: str) -> None:
    """刷新日志缓冲区到文件"""
    global _log_buffer
    
    if not _log_buffer:
        return
    
    try:
        with open(log_path, 'a', encoding='utf-8') as f:
            f.writelines(_log_buffer)
        _log_buffer = []
    except Exception as e:
        log_once('logfile_flush_err', f"Failed to flush log buffer: {e}")


def flush_logs(log_path: str = None) -> None:
    """强制刷新日志缓冲区"""
    if log_path:
        _flush_log_buffer(log_path)


def reset_log(log_path: str) -> None:
    """Reset the log file."""
    global _log_cleared, _log_buffer
    if _log_cleared:
        return
    try:
        import os
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"[MOE-HOOK] log reset at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"[MOE-HOOK] log level: {_LOG_LEVEL}\n")
        _log_cleared = True
        _log_buffer = []  # 清空缓冲区
        log_once('log_path', f"Hook log file: {log_path} (level={_LOG_LEVEL})")
    except Exception as e:
        log_once('log_reset_err', f"Failed to reset log file {log_path}: {e}")
