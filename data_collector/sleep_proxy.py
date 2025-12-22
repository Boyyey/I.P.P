"""
Sleep Proxy Tracker
Uses system activity patterns to infer sleep/wake cycles and circadian rhythms.
"""

import time
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import threading
import platform
import subprocess
import psutil


class SleepProxyTracker:
    def __init__(self, db_path: str = "data/performance_data.db"):
        self.db_path = db_path
        self.tracking = False
        self.tracking_thread = None
        self.last_activity_time = time.time()
        self.inactivity_threshold = 300  # 5 minutes of inactivity = potential sleep
        self._init_database()
    
    def _init_database(self):
        """Initialize database for sleep tracking."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sleep_proxy (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                is_active INTEGER,
                activity_type TEXT,
                session_duration REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sleep_periods (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sleep_start TEXT NOT NULL,
                sleep_end TEXT,
                duration_hours REAL,
                sleep_quality_score REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def check_system_activity(self) -> Dict:
        """Check if system is currently active."""
        try:
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Check mouse/keyboard activity (platform-specific)
            mouse_active = self._check_mouse_activity()
            keyboard_active = self._check_keyboard_activity()
            
            # Check for active processes
            active_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
                try:
                    if proc.info['cpu_percent'] > 1.0:  # Process using >1% CPU
                        active_processes.append(proc.info['name'])
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Determine overall activity
            is_active = (
                cpu_percent > 5.0 or  # CPU usage
                mouse_active or
                keyboard_active or
                len(active_processes) > 0
            )
            
            return {
                'timestamp': datetime.now().isoformat(),
                'is_active': int(is_active),
                'cpu_percent': cpu_percent,
                'mouse_active': mouse_active,
                'keyboard_active': keyboard_active,
                'active_processes': len(active_processes),
                'activity_type': self._determine_activity_type(cpu_percent, mouse_active, keyboard_active)
            }
        except Exception as e:
            print(f"Error checking system activity: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'is_active': 0,
                'activity_type': 'error'
            }
    
    def _check_mouse_activity(self) -> bool:
        """Check for recent mouse activity."""
        system = platform.system()
        
        try:
            if system == "Windows":
                return self._check_windows_mouse_activity()
            elif system == "Darwin":  # macOS
                return self._check_macos_mouse_activity()
            elif system == "Linux":
                return self._check_linux_mouse_activity()
        except Exception as e:
            print(f"Mouse activity check error: {e}")
        
        return False
    
    def _check_windows_mouse_activity(self) -> bool:
        """Check mouse activity on Windows."""
        try:
            script = '''
            Add-Type -TypeDefinition '
                using System;
                using System.Runtime.InteropServices;
                public class User32 {
                    [DllImport("user32.dll")]
                    public static extern bool GetLastInputInfo(ref LASTINPUTINFO plii);
                    
                    [StructLayout(LayoutKind.Sequential)]
                    public struct LASTINPUTINFO {
                        public static readonly int SizeOf = Marshal.SizeOf(typeof(LASTINPUTINFO));
                        public uint cbSize;
                        public uint dwTime;
                    }
                }
            '
            
            $lii = New-Object User32+LASTINPUTINFO
            $lii.cbSize = [User32+LASTINPUTINFO]::SizeOf
            $lii.dwTime = 0
            
            if ([User32]::GetLastInputInfo([ref]$lii)) {
                $idleTime = (Get-Date).AddMilliseconds(-$lii.dwTime).TotalMilliseconds
                $idleTime
            }
            '''
            
            result = subprocess.run(['powershell', '-Command', script], 
                                  capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0 and result.stdout.strip():
                idle_ms = float(result.stdout.strip())
                return idle_ms < 60000  # Active if idle time < 1 minute
        except Exception as e:
            print(f"Windows mouse activity error: {e}")
        
        return False
    
    def _check_macos_mouse_activity(self) -> bool:
        """Check mouse activity on macOS."""
        try:
            # Use ioreg to check for mouse/keyboard activity
            result = subprocess.run(['ioreg', '-c', 'IOHIDSystem'], 
                                  capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                # Parse for HIDIdleTime
                for line in result.stdout.split('\n'):
                    if 'HIDIdleTime' in line:
                        # Extract idle time value
                        parts = line.split('=')
                        if len(parts) > 1:
                            try:
                                idle_ns = int(parts[1].strip())
                                idle_ms = idle_ns / 1000000  # Convert nanoseconds to milliseconds
                                return idle_ms < 60000
                            except ValueError:
                                pass
        except Exception as e:
            print(f"macOS mouse activity error: {e}")
        
        return False
    
    def _check_linux_mouse_activity(self) -> bool:
        """Check mouse activity on Linux."""
        try:
            # Check for recent X11 activity
            result = subprocess.run(['xprintidle'], 
                                  capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0 and result.stdout.strip():
                idle_ms = int(result.stdout.strip())
                return idle_ms < 60000
        except Exception as e:
            print(f"Linux mouse activity error: {e}")
        
        return False
    
    def _check_keyboard_activity(self) -> bool:
        """Check for recent keyboard activity."""
        # For simplicity, we'll use the same idle time detection as mouse
        return self._check_mouse_activity()
    
    def _determine_activity_type(self, cpu_percent: float, mouse_active: bool, keyboard_active: bool) -> str:
        """Determine the type of activity."""
        if cpu_percent > 50:
            return 'heavy_computing'
        elif keyboard_active and mouse_active:
            return 'active_work'
        elif keyboard_active:
            return 'typing'
        elif mouse_active:
            return 'browsing'
        elif cpu_percent > 10:
            return 'background_task'
        else:
            return 'minimal'
    
    def store_activity(self, activity: Dict):
        """Store activity record in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO sleep_proxy 
            (timestamp, is_active, activity_type, session_duration)
            VALUES (?, ?, ?, ?)
        ''', (
            activity['timestamp'],
            activity['is_active'],
            activity['activity_type'],
            0  # Will be updated later
        ))
        
        conn.commit()
        conn.close()
    
    def detect_sleep_periods(self, hours: int = 48) -> List[Dict]:
        """Detect sleep periods from activity data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get recent activity data
        cursor.execute('''
            SELECT timestamp, is_active, activity_type
            FROM sleep_proxy 
            WHERE timestamp > datetime('now', '-{} hours')
            ORDER BY timestamp ASC
        '''.format(hours))
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return []
        
        sleep_periods = []
        in_sleep_period = False
        sleep_start = None
        
        for i, (timestamp, is_active, activity_type) in enumerate(rows):
            current_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            
            if is_active == 0 and not in_sleep_period:
                # Start of sleep period
                in_sleep_period = True
                sleep_start = current_time
            elif is_active == 1 and in_sleep_period:
                # End of sleep period
                sleep_end = current_time
                duration = (sleep_end - sleep_start).total_seconds() / 3600  # Convert to hours
                
                # Only consider periods longer than 1 hour as sleep
                if duration >= 1.0:
                    sleep_periods.append({
                        'sleep_start': sleep_start.isoformat(),
                        'sleep_end': sleep_end.isoformat(),
                        'duration_hours': duration,
                        'sleep_quality_score': self._calculate_sleep_quality(duration)
                    })
                
                in_sleep_period = False
        
        # Handle case where we're still in sleep period
        if in_sleep_period and sleep_start:
            sleep_end = datetime.now()
            duration = (sleep_end - sleep_start).total_seconds() / 3600
            
            if duration >= 1.0:
                sleep_periods.append({
                    'sleep_start': sleep_start.isoformat(),
                    'sleep_end': None,  # Still sleeping
                    'duration_hours': duration,
                    'sleep_quality_score': self._calculate_sleep_quality(duration)
                })
        
        return sleep_periods
    
    def _calculate_sleep_quality(self, duration_hours: float) -> float:
        """Calculate a simple sleep quality score based on duration."""
        # Optimal sleep duration: 7-9 hours
        if 7 <= duration_hours <= 9:
            return 1.0
        elif 6 <= duration_hours < 7 or 9 < duration_hours <= 10:
            return 0.8
        elif 5 <= duration_hours < 6 or 10 < duration_hours <= 11:
            return 0.6
        elif 4 <= duration_hours < 5 or 11 < duration_hours <= 12:
            return 0.4
        else:
            return 0.2  # Too little or too much sleep
    
    def start_tracking(self, interval: int = 60):
        """Start continuous sleep proxy tracking."""
        if self.tracking:
            print("Sleep proxy tracking already running")
            return
        
        self.tracking = True
        
        def tracking_loop():
            while self.tracking:
                activity = self.check_system_activity()
                self.store_activity(activity)
                
                if activity['is_active']:
                    self.last_activity_time = time.time()
                
                time.sleep(interval)
        
        self.tracking_thread = threading.Thread(target=tracking_loop, daemon=True)
        self.tracking_thread.start()
        print(f"Started sleep proxy tracking with {interval}s interval")
    
    def stop_tracking(self):
        """Stop sleep proxy tracking."""
        self.tracking = False
        if self.tracking_thread:
            self.tracking_thread.join(timeout=5)
        print("Stopped sleep proxy tracking")
    
    def get_circadian_metrics(self, days: int = 7) -> Dict:
        """Calculate circadian rhythm metrics."""
        sleep_periods = self.detect_sleep_periods(days * 24)
        
        if not sleep_periods:
            return {
                'period_days': days,
                'avg_sleep_duration': 0,
                'sleep_regularity': 0,
                'bedtime_consistency': 0,
                'sleep_quality_avg': 0,
                'total_sleep_periods': 0
            }
        
        # Calculate average sleep duration
        completed_sleeps = [s for s in sleep_periods if s['sleep_end']]
        if completed_sleeps:
            avg_sleep_duration = sum(s['duration_hours'] for s in completed_sleeps) / len(completed_sleeps)
        else:
            avg_sleep_duration = 0
        
        # Calculate sleep regularity (consistency of sleep duration)
        if len(completed_sleeps) > 1:
            durations = [s['duration_hours'] for s in completed_sleeps]
            avg_duration = sum(durations) / len(durations)
            variance = sum((d - avg_duration)**2 for d in durations) / len(durations)
            sleep_regularity = max(0, 1 - (variance / 4))  # Normalize to 0-1
        else:
            sleep_regularity = 0
        
        # Calculate bedtime consistency
        if len(completed_sleeps) > 1:
            bedtimes = []
            for sleep in completed_sleeps:
                sleep_start = datetime.fromisoformat(sleep['sleep_start'].replace('Z', '+00:00'))
                bedtime_hour = sleep_start.hour + sleep_start.minute / 60
                bedtimes.append(bedtime_hour)
            
            avg_bedtime = sum(bedtimes) / len(bedtimes)
            bedtime_variance = sum((b - avg_bedtime)**2 for b in bedtimes) / len(bedtimes)
            bedtime_consistency = max(0, 1 - (bedtime_variance / 9))  # Normalize to 0-1
        else:
            bedtime_consistency = 0
        
        # Calculate average sleep quality
        if sleep_periods:
            avg_quality = sum(s['sleep_quality_score'] for s in sleep_periods) / len(sleep_periods)
        else:
            avg_quality = 0
        
        return {
            'period_days': days,
            'avg_sleep_duration': avg_sleep_duration,
            'sleep_regularity': sleep_regularity,
            'bedtime_consistency': bedtime_consistency,
            'sleep_quality_avg': avg_quality,
            'total_sleep_periods': len(sleep_periods),
            'sleep_periods': sleep_periods
        }
    
    def get_fatigue_indicators(self, hours: int = 24) -> Dict:
        """Get fatigue indicators based on sleep patterns."""
        circadian = self.get_circadian_metrics(1)
        
        # Fatigue indicators
        sleep_debt = max(0, 8 - circadian['avg_sleep_duration'])  # Hours of missed sleep
        poor_regularity = circadian['sleep_regularity'] < 0.7
        low_quality = circadian['sleep_quality_avg'] < 0.6
        
        # Calculate fatigue score
        fatigue_score = (
            min(sleep_debt / 4, 0.4) +  # Sleep debt contribution
            (0.3 if poor_regularity else 0) +  # Irregular sleep contribution
            (0.3 if low_quality else 0)  # Poor quality contribution
        )
        
        return {
            'fatigue_score': min(fatigue_score, 1.0),
            'sleep_debt_hours': sleep_debt,
            'irregular_schedule': poor_regularity,
            'poor_sleep_quality': low_quality,
            'recommendation': self._get_sleep_recommendation(fatigue_score, circadian)
        }
    
    def _get_sleep_recommendation(self, score: float, circadian: Dict) -> str:
        """Get sleep recommendation based on fatigue score and patterns."""
        if score < 0.3:
            return "Sleep patterns look healthy"
        elif score < 0.6:
            return "Consider maintaining a more consistent sleep schedule"
        elif score < 0.8:
            return "Recommended: Prioritize 7-9 hours of sleep with consistent bedtime"
        else:
            return "High sleep fatigue detected - Focus on sleep hygiene and consistency"


if __name__ == "__main__":
    tracker = SleepProxyTracker()
    
    # Test tracking for 2 minutes
    print("Starting sleep proxy tracking for 2 minutes...")
    tracker.start_tracking(interval=30)
    
    time.sleep(120)
    
    tracker.stop_tracking()
    
    # Display metrics
    circadian = tracker.get_circadian_metrics(1)
    print("Circadian metrics:", json.dumps(circadian, indent=2, default=str))
    
    fatigue = tracker.get_fatigue_indicators(1)
    print("Fatigue indicators:", json.dumps(fatigue, indent=2))
