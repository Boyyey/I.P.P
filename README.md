# Intelligent Performance Predictor

An AI-powered system that monitors and predicts personal performance and burnout risk using non-invasive behavioral data.

## Features

- 📊 Real-time performance monitoring
- 🔥 Burnout risk assessment
- 🎯 Focus and energy level tracking
- 📈 Historical data analysis
- 🔔 Smart alerts and recommendations
- 📱 Responsive web dashboard

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/boyyey/intelligent-performance-predictor.git
   cd intelligent-performance-predictor
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Initialize the database:
   ```bash
   python -c "from data_collector.database import init_db; init_db()"
   ```

## Usage

1. Start the data collectors (in separate terminals):
   ```bash
   python -m data_collector.system_metrics
   python -m data_collector.app_usage
   python -m data_collector.typing_speed
   ```

2. Run the Streamlit dashboard:
   ```bash
   streamlit run dashboard/ui.py
   ```

3. Open your browser and navigate to `http://localhost:8501`

## Dashboard Features

### Dashboard View
- Real-time performance metrics
- Burnout risk assessment
- Focus and energy levels
- Interactive charts
- Recent alerts

### Analytics View
- Historical performance trends
- Daily and weekly patterns
- Correlation analysis
- Exportable reports

### Settings
- Alert preferences
- Data collection settings
- Notification controls

## Data Collection

The system collects the following data:

- System metrics (CPU, memory, disk usage)
- Application usage patterns
- Typing speed and accuracy
- Sleep and activity patterns (derived)
- Journal entries (optional)

## Privacy

All data is stored locally on your machine. No personal data is sent to external servers.

## License

MIT License - See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Support

For support, please open an issue in the GitHub repository.
