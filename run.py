# run.py
import os
import sys
from pathlib import Path

def main():
    # Ensure data directory exists
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Initialize database if it doesn't exist
    db_path = data_dir / 'performance_data.db'
    if not db_path.exists():
        from data_collector.database import init_db
        init_db()
        print("Database initialized successfully.")
    
    # Start the dashboard
    os.system('streamlit run dashboard/ui.py')

if __name__ == '__main__':
    main()