"""
Typing Speed and Error Rate Monitor
Tracks typing patterns, speed, and error rates as indicators of cognitive fatigue.
"""

import time
import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import threading
import pynput
from pynput import keyboard, mouse
import math


class TypingSpeedMonitor:
    def __init__(self, db_path: str = "data/performance_data.db"):
        self.db_path = db_path
        self.monitoring = False
        self.monitor_thread = None
        self.keyboard_listener = None
        self.mouse_listener = None
        
        # Tracking variables
        self.keystrokes = []
        self.current_session_start = None
        self.last_keystroke_time = None
        self.typing_sessions = []
        self.error_corrections = []
        self.mouse_clicks = []
        self.mouse_movements = []
        
        self._init_database()
    
    def _init_database(self):
        """Initialize database for typing metrics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS typing_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                session_id TEXT,
                keystrokes_count INTEGER,
                session_duration_seconds REAL,
                wpm REAL,
                error_rate REAL,
                avg_key_interval REAL,
                backspace_count INTEGER,
                typing_variance REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS mouse_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                session_id TEXT,
                click_count INTEGER,
                movement_distance REAL,
                avg_click_interval REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _on_key_press(self, key):
        """Handle keyboard press events."""
        current_time = time.time()
        
        # Start new session if this is the first keystroke or after a long pause
        if (self.current_session_start is None or 
            (self.last_keystroke_time and current_time - self.last_keystroke_time > 10)):
            
            # Save previous session if exists
            if self.keystrokes:
                self._save_typing_session()
            
            # Start new session
            self.current_session_start = current_time
            self.keystrokes = []
        
        # Record keystroke
        try:
            key_char = key.char
            is_backspace = False
        except AttributeError:
            key_char = str(key)
            is_backspace = key == keyboard.Key.backspace
        
        self.keystrokes.append({
            'timestamp': current_time,
            'key': key_char,
            'is_backspace': is_backspace
        })
        
        self.last_keystroke_time = current_time
    
    def _on_key_release(self, key):
        """Handle keyboard release events."""
        pass
    
    def _on_mouse_click(self, x, y, button, pressed):
        """Handle mouse click events."""
        if pressed:
            current_time = time.time()
            self.mouse_clicks.append({
                'timestamp': current_time,
                'x': x,
                'y': y,
                'button': str(button)
            })
    
    def _on_mouse_move(self, x, y):
        """Handle mouse movement events."""
        current_time = time.time()
        
        # Only track movements periodically to avoid too much data
        if (not self.mouse_movements or 
            current_time - self.mouse_movements[-1]['timestamp'] > 0.1):
            
            if self.mouse_movements:
                last_pos = self.mouse_movements[-1]
                distance = math.sqrt((x - last_pos['x'])**2 + (y - last_pos['y'])**2)
                self.mouse_movements.append({
                    'timestamp': current_time,
                    'x': x,
                    'y': y,
                    'distance': distance
                })
            else:
                self.mouse_movements.append({
                    'timestamp': current_time,
                    'x': x,
                    'y': y,
                    'distance': 0
                })
    
    def _calculate_typing_metrics(self, keystrokes: List[Dict]) -> Dict:
        """Calculate typing metrics from keystroke data."""
        if len(keystrokes) < 2:
            return {
                'wpm': 0,
                'error_rate': 0,
                'avg_key_interval': 0,
                'backspace_count': 0,
                'typing_variance': 0
            }
        
        # Calculate session duration
        start_time = keystrokes[0]['timestamp']
        end_time = keystrokes[-1]['timestamp']
        duration = end_time - start_time
        
        if duration <= 0:
            return {
                'wpm': 0,
                'error_rate': 0,
                'avg_key_interval': 0,
                'backspace_count': 0,
                'typing_variance': 0
            }
        
        # Count characters (excluding backspaces)
        char_count = sum(1 for k in keystrokes if not k['is_backspace'])
        backspace_count = sum(1 for k in keystrokes if k['is_backspace'])
        
        # Calculate WPM (assuming average word length of 5 characters)
        wpm = (char_count / 5) / (duration / 60) if duration > 0 else 0
        
        # Calculate error rate
        total_keystrokes = len(keystrokes)
        error_rate = backspace_count / total_keystrokes if total_keystrokes > 0 else 0
        
        # Calculate average key interval
        intervals = []
        for i in range(1, len(keystrokes)):
            interval = keystrokes[i]['timestamp'] - keystrokes[i-1]['timestamp']
            intervals.append(interval)
        
        avg_key_interval = sum(intervals) / len(intervals) if intervals else 0
        
        # Calculate typing variance (consistency of typing speed)
        if len(intervals) > 1:
            mean_interval = avg_key_interval
            variance = sum((x - mean_interval)**2 for x in intervals) / len(intervals)
            typing_variance = math.sqrt(variance)  # Standard deviation
        else:
            typing_variance = 0
        
        return {
            'wpm': wpm,
            'error_rate': error_rate,
            'avg_key_interval': avg_key_interval,
            'backspace_count': backspace_count,
            'typing_variance': typing_variance
        }
    
    def _save_typing_session(self):
        """Save current typing session to database."""
        if not self.keystrokes:
            return
        
        metrics = self._calculate_typing_metrics(self.keystrokes)
        
        session_id = f"session_{int(self.current_session_start)}"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO typing_metrics 
            (timestamp, session_id, keystrokes_count, session_duration_seconds,
             wpm, error_rate, avg_key_interval, backspace_count, typing_variance)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.fromtimestamp(self.current_session_start).isoformat(),
            session_id,
            len(self.keystrokes),
            self.keystrokes[-1]['timestamp'] - self.keystrokes[0]['timestamp'],
            metrics['wpm'],
            metrics['error_rate'],
            metrics['avg_key_interval'],
            metrics['backspace_count'],
            metrics['typing_variance']
        ))
        
        conn.commit()
        conn.close()
    
    def _save_mouse_session(self):
        """Save mouse tracking session to database."""
        if not self.mouse_clicks:
            return
        
        session_id = f"mouse_session_{int(time.time())}"
        
        # Calculate total movement distance
        total_distance = sum(m['distance'] for m in self.mouse_movements)
        
        # Calculate click intervals
        click_intervals = []
        for i in range(1, len(self.mouse_clicks)):
            interval = self.mouse_clicks[i]['timestamp'] - self.mouse_clicks[i-1]['timestamp']
            click_intervals.append(interval)
        
        avg_click_interval = sum(click_intervals) / len(click_intervals) if click_intervals else 0
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO mouse_metrics 
            (timestamp, session_id, click_count, movement_distance, avg_click_interval)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            session_id,
            len(self.mouse_clicks),
            total_distance,
            avg_click_interval
        ))
        
        conn.commit()
        conn.close()
    
    def start_monitoring(self):
        """Start typing and mouse monitoring."""
        if self.monitoring:
            print("Typing monitoring already running")
            return
        
        self.monitoring = True
        self.keystrokes = []
        self.mouse_clicks = []
        self.mouse_movements = []
        
        # Start keyboard listener
        self.keyboard_listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release
        )
        
        # Start mouse listener
        self.mouse_listener = mouse.Listener(
            on_click=self._on_mouse_click,
            on_move=self._on_mouse_move
        )
        
        self.keyboard_listener.start()
        self.mouse_listener.start()
        
        print("Started typing and mouse monitoring")
    
    def stop_monitoring(self):
        """Stop typing and mouse monitoring."""
        if not self.monitoring:
            return
        
        self.monitoring = False
        
        # Save final session
        if self.keystrokes:
            self._save_typing_session()
        
        if self.mouse_clicks:
            self._save_mouse_session()
        
        # Stop listeners
        if self.keyboard_listener:
            self.keyboard_listener.stop()
        
        if self.mouse_listener:
            self.mouse_listener.stop()
        
        print("Stopped typing and mouse monitoring")
    
    def get_typing_stats(self, hours: int = 24) -> Dict:
        """Get typing statistics for the last N hours."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT AVG(wpm), AVG(error_rate), AVG(avg_key_interval),
                   SUM(backspace_count), AVG(typing_variance),
                   COUNT(*) as session_count
            FROM typing_metrics 
            WHERE timestamp > datetime('now', '-{} hours')
        '''.format(hours))
        
        row = cursor.fetchone()
        conn.close()
        
        if row[6] == 0:  # session_count
            return {
                'period_hours': hours,
                'avg_wpm': 0,
                'avg_error_rate': 0,
                'avg_key_interval': 0,
                'total_backspaces': 0,
                'avg_typing_variance': 0,
                'session_count': 0
            }
        
        return {
            'period_hours': hours,
            'avg_wpm': row[0] or 0,
            'avg_error_rate': row[1] or 0,
            'avg_key_interval': row[2] or 0,
            'total_backspaces': row[3] or 0,
            'avg_typing_variance': row[4] or 0,
            'session_count': row[6] or 0
        }
    
    def get_fatigue_indicators(self, hours: int = 24) -> Dict:
        """Calculate fatigue indicators from typing patterns."""
        stats = self.get_typing_stats(hours)
        
        # Fatigue indicators
        speed_decline = 0  # Would need historical comparison
        error_increase = stats['avg_error_rate'] > 0.15  # High error rate
        variance_increase = stats['avg_typing_variance'] > 0.1  # Inconsistent typing
        
        # Calculate fatigue score (0-1, higher = more fatigued)
        fatigue_score = (
            min(stats['avg_error_rate'] * 2, 0.4) +  # Error rate contribution
            min(stats['avg_typing_variance'] * 5, 0.3) +  # Variance contribution
            (0.3 if stats['avg_wpm'] < 20 else 0)  # Low speed contribution
        )
        
        return {
            'fatigue_score': min(fatigue_score, 1.0),
            'speed_degradation': stats['avg_wpm'] < 30,
            'error_spike': error_increase,
            'inconsistent_typing': variance_increase,
            'recommendation': self._get_fatigue_recommendation(fatigue_score)
        }
    
    def _get_fatigue_recommendation(self, score: float) -> str:
        """Get recommendation based on fatigue score."""
        if score < 0.3:
            return "Typing patterns look healthy"
        elif score < 0.6:
            return "Consider taking a short break"
        elif score < 0.8:
            return "Recommended: Take a 15-20 minute break"
        else:
            return "High fatigue detected - Take a break immediately"


if __name__ == "__main__":
    monitor = TypingSpeedMonitor()
    
    print("Starting typing speed monitoring for 30 seconds...")
    print("Type something to test the monitoring!")
    
    monitor.start_monitoring()
    time.sleep(30)
    monitor.stop_monitoring()
    
    # Display stats
    stats = monitor.get_typing_stats(1)
    print("Typing stats:", json.dumps(stats, indent=2, default=str))
    
    fatigue = monitor.get_fatigue_indicators(1)
    print("Fatigue indicators:", json.dumps(fatigue, indent=2))
