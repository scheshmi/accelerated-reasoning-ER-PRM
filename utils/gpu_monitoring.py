import os
import logging
import time
import threading
import csv
from datetime import datetime
import GPUtil

logger = logging.getLogger(__name__)

gpu_monitoring_stop_flag = False

def monitor_gpu_usage(output_dir, interval=1.0):
    """
    Background thread function to monitor GPU usage at regular intervals
    
    Args:
        output_dir: Directory to save the GPU monitoring CSV file
        interval: Time interval between measurements in seconds
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = os.path.join(output_dir, f"gpu_monitoring_{timestamp}.csv")
    
    headers = ['timestamp', 'gpu_id', 'gpu_name', 'utilization_gpu', 'utilization_memory', 
               'memory_total', 'memory_used', 'memory_free', 'temperature']
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        while not gpu_monitoring_stop_flag:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            
            try:
                gpus = GPUtil.getGPUs()
                
                for gpu in gpus:
                    row = [
                        timestamp,
                        gpu.id,
                        gpu.name,
                        gpu.load * 100,  
                        gpu.memoryUtil * 100,  
                        gpu.memoryTotal,
                        gpu.memoryUsed,
                        gpu.memoryFree,
                        gpu.temperature
                    ]
                    writer.writerow(row)
                    
                f.flush()
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in GPU monitoring: {str(e)}")
                time.sleep(interval)  

def start_gpu_monitoring(output_dir, interval=1.0):
    """
    Start the GPU monitoring in a background thread
    
    Args:
        output_dir: Directory to save the monitoring data
        interval: Time interval between measurements in seconds
        
    Returns:
        threading.Thread: The monitoring thread object
    """
    global gpu_monitoring_stop_flag
    gpu_monitoring_stop_flag = False
    
    monitor_thread = threading.Thread(
        target=monitor_gpu_usage, 
        args=(output_dir, interval),
        daemon=True  
    )
    monitor_thread.start()
    logger.info(f"GPU monitoring started, logging to {output_dir} at {interval}s intervals")
    return monitor_thread

def stop_gpu_monitoring(monitor_thread):
    """
    Stop the GPU monitoring thread
    
    Args:
        monitor_thread: The monitoring thread to stop
    """
    global gpu_monitoring_stop_flag
    gpu_monitoring_stop_flag = True
    
    if monitor_thread and monitor_thread.is_alive():
        monitor_thread.join(timeout=5.0)
        logger.info("GPU monitoring stopped")

def get_system_info():
    """Get system GPU information."""
    gpu_info = []
    try:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            gpu_info.append({
                'id': gpu.id,
                'name': gpu.name,
                'memory_total': gpu.memoryTotal,
                'memory_used': gpu.memoryUsed,
                'memory_free': gpu.memoryFree,
                'temperature': gpu.temperature,
                'uuid': gpu.uuid
            })
    except Exception as e:
        logger.error(f"Error getting GPU info: {str(e)}")
    return gpu_info

