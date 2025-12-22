"""
Application Usage Tracker
Monitors application switching frequency and duration as indicators of focus and cognitive load.
"""

import time
import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional, Set
import threading
import platform
import subprocess
import psutil


class AppUsageTracker:
    def __init__(self, db_path: str = "data/performance_data.db"):
        self.db_path = db_path
        self.collecting = False
        self.collection_thread = None
        self.current_app = None
        self.app_start_time = None
        self.session_start = datetime.now()
        self._init_database()
    
    def _init_database(self):
        """Initialize database for app usage tracking."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS app_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                app_name TEXT NOT NULL,
                window_title TEXT,
                duration_seconds REAL,
                session_duration REAL,
                is_switch INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS app_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_start TEXT NOT NULL,
                session_end TEXT,
                total_switches INTEGER DEFAULT 0,
                unique_apps INTEGER DEFAULT 0,
                total_duration REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_active_window_info(self) -> Optional[Dict]:
        """Get information about the currently active window/application."""
        system = platform.system()
        
        try:
            if system == "Windows":
                return self._get_windows_active_window()
            elif system == "Darwin":  # macOS
                return self._get_macos_active_window()
            elif system == "Linux":
                return self._get_linux_active_window()
            else:
                return None
        except Exception as e:
            print(f"Error getting active window: {e}")
            return None
    
    def _get_windows_active_window(self) -> Optional[Dict]:
        """Get active window on Windows using PowerShell."""
        try:
            script = '''
            Add-Type -TypeDefinition '
                using System;
                using System.Runtime.InteropServices;
                public class User32 {
                    [DllImport("user32.dll")]
                    public static extern IntPtr GetForegroundWindow();
                    
                    [DllImport("user32.dll")]
                    public static extern int GetWindowText(IntPtr hWnd, System.Text.StringBuilder text, int count);
                    
                    [DllImport("user32.dll")]
                    public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint lpdwProcessId);
                }
            '
            
            $hWnd = [User32]::GetForegroundWindow()
            $processId = 0
            [User32]::GetWindowThreadProcessId($hWnd, [ref]$processId)
            $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
            
            if ($process) {
                $title = New-Object System.Text.StringBuilder 256
                [User32]::GetWindowText($hWnd, $title, 256) | Out-Null
                
                @{
                    "app_name" = $process.ProcessName
                    "window_title" = $title.ToString()
                    "process_id" = $processId
                    "executable" = $process.Path
                }
            }
            '''
            
            result = subprocess.run(['powershell', '-Command', script], 
                                  capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0 and result.stdout.strip():
                # Parse PowerShell output
                lines = result.stdout.strip().split('\n')
                data = {}
                for line in lines:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        data[key.strip()] = value.strip().strip('"')
                
                return {
                    'app_name': data.get('app_name', 'Unknown'),
                    'window_title': data.get('window_title', ''),
                    'process_id': int(data.get('process_id', 0)),
                    'executable': data.get('executable', '')
                }
        except Exception as e:
            print(f"Windows active window detection error: {e}")
        
        return None
    
    def _get_macos_active_window(self) -> Optional[Dict]:
        """Get active window on macOS."""
        try:
            script = '''
            tell application "System Events"
                set frontApp to name of first application process whose frontmost is true
                set frontWindow to name of front window of application process frontApp
            end tell
            
            do shell script "echo " & quoted form of frontApp & ":" & quoted form of frontWindow
            '''
            
            result = subprocess.run(['osascript', '-e', script], 
                                  capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0 and ':' in result.stdout:
                app_name, window_title = result.stdout.strip().split(':', 1)
                return {
                    'app_name': app_name,
                    'window_title': window_title,
                    'process_id': 0,
                    'executable': ''
                }
        except Exception as e:
            print(f"macOS active window detection error: {e}")
        
        return None
    
    def _get_linux_active_window(self) -> Optional[Dict]:
        """Get active window on Linux using xdotool."""
        try:
            # Get active window ID
            result = subprocess.run(['xdotool', 'getactivewindow'], 
                                  capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                window_id = result.stdout.strip()
                
                # Get window name
                result = subprocess.run(['xdotool', 'getwindowname', window_id], 
                                      capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0:
                    window_title = result.stdout.strip()
                    
                    # Get process info
                    result = subprocess.run(['xdotool', 'getwindowpid', window_id], 
                                          capture_output=True, text=True, timeout=5)
                    
                    if result.returncode == 0:
                        try:
                            pid = int(result.stdout.strip())
                            process = psutil.Process(pid)
                            
                            return {
                                'app_name': process.name(),
                                'window_title': window_title,
                                'process_id': pid,
                                'executable': process.exe()
                            }
                        except psutil.NoSuchProcess:
                            pass
                    
                    return {
                        'app_name': 'Unknown',
                        'window_title': window_title,
                        'process_id': 0,
                        'executable': ''
                    }
        except Exception as e:
            print(f"Linux active window detection error: {e}")
        
        return None
    
    def track_app_switch(self, current_time: datetime):
        """Track application switch and store usage data."""
        window_info = self.get_active_window_info()
        
        if not window_info:
            return
        
        app_name = window_info['app_name']
        window_title = window_info.get('window_title', '')
        
        # Check if this is an app switch
        is_switch = self.current_app != app_name
        
        if is_switch and self.current_app and self.app_start_time:
            # Calculate duration of previous app
            duration = (current_time - self.app_start_time).total_seconds()
            session_duration = (current_time - self.session_start).total_seconds()
            
            # Store previous app usage
            self._store_app_usage(
                timestamp=self.app_start_time.isoformat(),
                app_name=self.current_app,
                window_title='',
                duration_seconds=duration,
                session_duration=session_duration,
                is_switch=1
            )
        
        # Update current app tracking
        self.current_app = app_name
        self.app_start_time = current_time
    
    def _store_app_usage(self, timestamp: str, app_name: str, window_title: str,
                        duration_seconds: float, session_duration: float, is_switch: int):
        """Store app usage record in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO app_usage 
            (timestamp, app_name, window_title, duration_seconds, session_duration, is_switch)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (timestamp, app_name, window_title, duration_seconds, session_duration, is_switch))
        
        conn.commit()
        conn.close()
    
    def start_tracking(self, interval: int = 5):
        """Start continuous app usage tracking."""
        if self.collecting:
            print("App tracking already running")
            return
        
        self.collecting = True
        self.session_start = datetime.now()
        
        def tracking_loop():
            while self.collecting:
                current_time = datetime.now()
                self.track_app_switch(current_time)
                time.sleep(interval)
            
            # End session
            self._end_session()
        
        self.collection_thread = threading.Thread(target=tracking_loop, daemon=True)
        self.collection_thread.start()
        print(f"Started app usage tracking with {interval}s interval")
    
    def stop_tracking(self):
        """Stop app usage tracking."""
        self.collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        print("Stopped app usage tracking")
    
    def _end_session(self):
        """End current tracking session."""
        if self.current_app and self.app_start_time:
            current_time = datetime.now()
            duration = (current_time - self.app_start_time).total_seconds()
            session_duration = (current_time - self.session_start).total_seconds()
            
            self._store_app_usage(
                timestamp=self.app_start_time.isoformat(),
                app_name=self.current_app,
                window_title='',
                duration_seconds=duration,
                session_duration=session_duration,
                is_switch=0
            )
    
    def get_usage_stats(self, hours: int = 24) -> Dict:
        """Get app usage statistics for the last N hours."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT app_name, COUNT(*) as switch_count, 
                   SUM(duration_seconds) as total_duration,
                   AVG(duration_seconds) as avg_duration
            FROM app_usage 
            WHERE timestamp > datetime('now', '-{} hours')
            GROUP BY app_name
            ORDER BY total_duration DESC
        '''.format(hours))
        
        rows = cursor.fetchall()
        conn.close()
        
        apps = []
        for row in rows:
            apps.append({
                'app_name': row[0],
                'switch_count': row[1],
                'total_duration': row[2] or 0,
                'avg_duration': row[3] or 0
            })
        
        # Calculate overall stats
        total_switches = sum(app['switch_count'] for app in apps)
        total_duration = sum(app['total_duration'] for app in apps)
        
        return {
            'period_hours': hours,
            'total_switches': total_switches,
            'total_duration': total_duration,
            'unique_apps': len(apps),
            'switch_rate': total_switches / (total_duration / 3600) if total_duration > 0 else 0,
            'apps': apps
        }
    
    def get_focus_metrics(self, hours: int = 24) -> Dict:
        """Calculate focus-related metrics."""
        stats = self.get_usage_stats(hours)
        
        # Focus duration (time spent in top 3 apps)
        top_apps = stats['apps'][:3]
        focus_time = sum(app['total_duration'] for app in top_apps)
        focus_ratio = focus_time / stats['total_duration'] if stats['total_duration'] > 0 else 0
        
        # App switching entropy (measure of fragmentation)
        if stats['total_duration'] > 0:
            probabilities = [app['total_duration'] / stats['total_duration'] for app in stats['apps']]
            entropy = -sum(p * (p and math.log(p) or 0) for p in probabilities)
        else:
            entropy = 0
        
        return {
            'focus_time_hours': focus_time / 3600,
            'focus_ratio': focus_ratio,
            'switch_entropy': entropy,
            'app_fragmentation': len(stats['apps']) / (stats['total_duration'] / 3600) if stats['total_duration'] > 0 else 0
        }


if __name__ == "__main__":
    import math
    
    tracker = AppUsageTracker()
    
    # Test tracking for 30 seconds
    print("Starting app usage tracking for 30 seconds...")
    tracker.start_tracking(interval=2)
    
    time.sleep(30)
    
    tracker.stop_tracking()
    
    # Display stats
    stats = tracker.get_usage_stats(1)
    print("Usage stats:", json.dumps(stats, indent=2, default=str))
    
    focus_metrics = tracker.get_focus_metrics(1)
    print("Focus metrics:", json.dumps(focus_metrics, indent=2))
