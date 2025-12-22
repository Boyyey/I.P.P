"""
System Metrics Collector
Collects CPU usage, memory, and other system performance metrics as proxy for cognitive load.
"""

import psutil
import time
import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional
import threading


class SystemMetricsCollector:
    def __init__(self, db_path: str = "data/performance_data.db"):
        self.db_path = db_path
        self.collecting = False
        self.collection_thread = None
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for storing metrics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                cpu_percent REAL,
                memory_percent REAL,
                disk_usage_percent REAL,
                active_processes INTEGER,
                battery_percent REAL,
                power_plugged INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def collect_metrics(self) -> Dict:
        """Collect current system metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage_percent = disk.percent
            
            # Process count
            active_processes = len(psutil.pids())
            
            # Battery info (if available)
            battery_percent = None
            power_plugged = None
            try:
                battery = psutil.sensors_battery()
                if battery:
                    battery_percent = battery.percent
                    power_plugged = int(battery.power_plugged)
            except:
                pass
            
            return {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'disk_usage_percent': disk_usage_percent,
                'active_processes': active_processes,
                'battery_percent': battery_percent,
                'power_plugged': power_plugged
            }
        except Exception as e:
            print(f"Error collecting system metrics: {e}")
            return None
    
    def store_metrics(self, metrics: Dict):
        """Store metrics in database."""
        if not metrics:
            return
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO system_metrics 
            (timestamp, cpu_percent, memory_percent, disk_usage_percent, 
             active_processes, battery_percent, power_plugged)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics['timestamp'],
            metrics['cpu_percent'],
            metrics['memory_percent'],
            metrics['disk_usage_percent'],
            metrics['active_processes'],
            metrics['battery_percent'],
            metrics['power_plugged']
        ))
        
        conn.commit()
        conn.close()
    
    def start_collection(self, interval: int = 60):
        """Start continuous metrics collection."""
        if self.collecting:
            print("Collection already running")
            return
        
        self.collecting = True
        
        def collect_loop():
            while self.collecting:
                metrics = self.collect_metrics()
                if metrics:
                    self.store_metrics(metrics)
                time.sleep(interval)
        
        self.collection_thread = threading.Thread(target=collect_loop, daemon=True)
        self.collection_thread.start()
        print(f"Started system metrics collection with {interval}s interval")
    
    def stop_collection(self):
        """Stop metrics collection."""
        self.collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        print("Stopped system metrics collection")
    
    def get_recent_metrics(self, hours: int = 24) -> List[Dict]:
        """Get metrics from the last N hours."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM system_metrics 
            WHERE timestamp > datetime('now', '-{} hours')
            ORDER BY timestamp DESC
        '''.format(hours))
        
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
        
        conn.close()
        
        return [dict(zip(columns, row)) for row in rows]
    
    def get_metrics_summary(self, hours: int = 24) -> Dict:
        """Get summary statistics for recent metrics."""
        metrics = self.get_recent_metrics(hours)
        if not metrics:
            return {}
        
        cpu_values = [m['cpu_percent'] for m in metrics if m['cpu_percent']]
        memory_values = [m['memory_percent'] for m in metrics if m['memory_percent']]
        
        summary = {
            'period_hours': hours,
            'sample_count': len(metrics),
            'cpu': {
                'avg': sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                'max': max(cpu_values) if cpu_values else 0,
                'min': min(cpu_values) if cpu_values else 0
            },
            'memory': {
                'avg': sum(memory_values) / len(memory_values) if memory_values else 0,
                'max': max(memory_values) if memory_values else 0,
                'min': min(memory_values) if memory_values else 0
            }
        }
        
        return summary


if __name__ == "__main__":
    # Test the collector
    collector = SystemMetricsCollector()
    
    # Collect and display current metrics
    metrics = collector.collect_metrics()
    print("Current metrics:", json.dumps(metrics, indent=2))
    
    # Store metrics
    collector.store_metrics(metrics)
    
    # Get recent metrics summary
    summary = collector.get_metrics_summary(1)
    print("Recent summary:", json.dumps(summary, indent=2))
