"""
Database initialization and management for the Intelligent Performance Predictor.
"""

import os
import sqlite3
from pathlib import Path

def init_db():
    """Initialize the SQLite database with required tables."""
    # Create data directory if it doesn't exist
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    db_path = data_dir / 'performance_data.db'
    
    # Connect to SQLite database (creates it if it doesn't exist)
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    try:
        # Create system_metrics table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            cpu_percent REAL,
            memory_percent REAL,
            disk_percent REAL,
            battery_percent REAL,
            active_processes INTEGER
        )
        ''')
        
        # Create app_usage table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS app_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            window_title TEXT,
            process_name TEXT,
            duration_seconds INTEGER
        )
        ''')
        
        # Create typing_metrics table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS typing_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            wpm REAL,
            error_rate REAL,
            session_duration_seconds INTEGER
        )
        ''')
        
        # Create performance_metrics table (for ML features)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS performance_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            performance_score REAL,
            burnout_risk REAL,
            focus_level REAL,
            energy_level REAL,
            stress_level REAL,
            activity_intensity REAL
        )
        ''')
        
        # Create alerts table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id TEXT PRIMARY KEY,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            type TEXT,
            severity TEXT,
            title TEXT,
            message TEXT,
            acknowledged BOOLEAN DEFAULT 0,
            resolved BOOLEAN DEFAULT 0
        )
        ''')
        
        # Create indexes for better query performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_app_usage_timestamp ON app_usage(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_typing_metrics_timestamp ON typing_metrics(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_performance_metrics_timestamp ON performance_metrics(timestamp)')
        
        conn.commit()
        print(f"Database initialized successfully at {db_path}")
        
    except Exception as e:
        print(f"Error initializing database: {e}")
        conn.rollback()
    finally:
        conn.close()

def get_db_connection():
    """Get a connection to the SQLite database."""
    db_path = Path('data') / 'performance_data.db'
    return sqlite3.connect(str(db_path))

if __name__ == "__main__":
    init_db()
