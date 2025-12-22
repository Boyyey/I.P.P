"""
Journaling and Sentiment Analysis Module
Optional journaling system with sentiment analysis to track mood and stress levels.
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import re
import textblob
from textblob import TextBlob
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os


class JournalingSystem:
    def __init__(self, db_path: str = "data/performance_data.db"):
        self.db_path = db_path
        self.sentiment_analyzer = None
        self._init_database()
        self._init_sentiment_analysis()
    
    def _init_database(self):
        """Initialize database for journaling."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS journal_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                entry_text TEXT NOT NULL,
                mood_rating INTEGER,
                stress_level INTEGER,
                energy_level INTEGER,
                productivity_rating INTEGER,
                sentiment_score REAL,
                sentiment_label TEXT,
                keywords TEXT,
                word_count INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS mood_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                avg_mood REAL,
                avg_stress REAL,
                avg_energy REAL,
                avg_productivity REAL,
                sentiment_trend REAL,
                entry_count INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _init_sentiment_analysis(self):
        """Initialize sentiment analysis tools."""
        try:
            # Download NLTK data if needed
            try:
                nltk.data.find('sentiment/vader_lexicon.zip')
            except LookupError:
                nltk.download('vader_lexicon')
            
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        except Exception as e:
            print(f"Warning: Could not initialize sentiment analysis: {e}")
            self.sentiment_analyzer = None
    
    def add_entry(self, entry_text: str, mood_rating: Optional[int] = None,
                  stress_level: Optional[int] = None, energy_level: Optional[int] = None,
                  productivity_rating: Optional[int] = None) -> Dict:
        """Add a new journal entry with optional ratings."""
        
        # Clean and validate entry text
        entry_text = entry_text.strip()
        if not entry_text:
            raise ValueError("Entry text cannot be empty")
        
        # Perform sentiment analysis
        sentiment_score, sentiment_label = self._analyze_sentiment(entry_text)
        
        # Extract keywords
        keywords = self._extract_keywords(entry_text)
        
        # Count words
        word_count = len(entry_text.split())
        
        # Store entry
        entry_data = {
            'timestamp': datetime.now().isoformat(),
            'entry_text': entry_text,
            'mood_rating': mood_rating,
            'stress_level': stress_level,
            'energy_level': energy_level,
            'productivity_rating': productivity_rating,
            'sentiment_score': sentiment_score,
            'sentiment_label': sentiment_label,
            'keywords': json.dumps(keywords),
            'word_count': word_count
        }
        
        self._store_entry(entry_data)
        
        return entry_data
    
    def _analyze_sentiment(self, text: str) -> Tuple[float, str]:
        """Analyze sentiment of text."""
        try:
            if self.sentiment_analyzer:
                # Use VADER for more nuanced sentiment analysis
                scores = self.sentiment_analyzer.polarity_scores(text)
                compound_score = scores['compound']
                
                # Determine sentiment label
                if compound_score >= 0.05:
                    label = 'positive'
                elif compound_score <= -0.05:
                    label = 'negative'
                else:
                    label = 'neutral'
                
                return compound_score, label
            else:
                # Fallback to TextBlob
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                
                if polarity > 0.1:
                    label = 'positive'
                elif polarity < -0.1:
                    label = 'negative'
                else:
                    label = 'neutral'
                
                return polarity, label
        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            return 0.0, 'neutral'
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text."""
        try:
            # Simple keyword extraction using TextBlob
            blob = TextBlob(text)
            
            # Get noun phrases and important words
            noun_phrases = blob.noun_phrases
            
            # Get individual words and filter
            words = blob.words.lower()
            
            # Filter out common stop words and short words
            stop_words = {'the', 'is', 'at', 'which', 'on', 'and', 'a', 'an', 'as', 'are', 'was', 
                         'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                         'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that',
                         'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
                         'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their'}
            
            keywords = []
            
            # Add noun phrases
            for phrase in noun_phrases:
                if len(phrase) > 2 and phrase not in stop_words:
                    keywords.append(phrase)
            
            # Add individual words
            for word in words:
                if (len(word) > 3 and 
                    word not in stop_words and 
                    word.isalpha() and 
                    word not in keywords):
                    keywords.append(word)
            
            # Limit to top 10 keywords
            return keywords[:10]
            
        except Exception as e:
            print(f"Keyword extraction error: {e}")
            return []
    
    def _store_entry(self, entry_data: Dict):
        """Store journal entry in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO journal_entries 
            (timestamp, entry_text, mood_rating, stress_level, energy_level,
             productivity_rating, sentiment_score, sentiment_label, keywords, word_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            entry_data['timestamp'],
            entry_data['entry_text'],
            entry_data['mood_rating'],
            entry_data['stress_level'],
            entry_data['energy_level'],
            entry_data['productivity_rating'],
            entry_data['sentiment_score'],
            entry_data['sentiment_label'],
            entry_data['keywords'],
            entry_data['word_count']
        ))
        
        conn.commit()
        conn.close()
    
    def get_recent_entries(self, days: int = 7, limit: int = 50) -> List[Dict]:
        """Get recent journal entries."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM journal_entries 
            WHERE timestamp > datetime('now', '-{} days')
            ORDER BY timestamp DESC
            LIMIT ?
        '''.format(days), (limit,))
        
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
        
        conn.close()
        
        entries = []
        for row in rows:
            entry = dict(zip(columns, row))
            # Parse keywords JSON
            if entry['keywords']:
                try:
                    entry['keywords'] = json.loads(entry['keywords'])
                except:
                    entry['keywords'] = []
            else:
                entry['keywords'] = []
            entries.append(entry)
        
        return entries
    
    def get_mood_trends(self, days: int = 30) -> Dict:
        """Analyze mood trends over time."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT DATE(timestamp) as date,
                   AVG(mood_rating) as avg_mood,
                   AVG(stress_level) as avg_stress,
                   AVG(energy_level) as avg_energy,
                   AVG(productivity_rating) as avg_productivity,
                   AVG(sentiment_score) as avg_sentiment,
                   COUNT(*) as entry_count
            FROM journal_entries 
            WHERE timestamp > datetime('now', '-{} days')
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
        '''.format(days))
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return {
                'period_days': days,
                'daily_averages': [],
                'overall_trends': {
                    'mood_trend': 0,
                    'stress_trend': 0,
                    'energy_trend': 0,
                    'productivity_trend': 0,
                    'sentiment_trend': 0
                }
            }
        
        daily_averages = []
        for row in rows:
            daily_averages.append({
                'date': row[0],
                'avg_mood': row[1] or 0,
                'avg_stress': row[2] or 0,
                'avg_energy': row[3] or 0,
                'avg_productivity': row[4] or 0,
                'avg_sentiment': row[5] or 0,
                'entry_count': row[6] or 0
            })
        
        # Calculate trends (simple linear trend over the period)
        if len(daily_averages) > 1:
            mood_trend = self._calculate_trend([d['avg_mood'] for d in daily_averages if d['avg_mood'] > 0])
            stress_trend = self._calculate_trend([d['avg_stress'] for d in daily_averages if d['avg_stress'] > 0])
            energy_trend = self._calculate_trend([d['avg_energy'] for d in daily_averages if d['avg_energy'] > 0])
            productivity_trend = self._calculate_trend([d['avg_productivity'] for d in daily_averages if d['avg_productivity'] > 0])
            sentiment_trend = self._calculate_trend([d['avg_sentiment'] for d in daily_averages])
        else:
            mood_trend = stress_trend = energy_trend = productivity_trend = sentiment_trend = 0
        
        return {
            'period_days': days,
            'daily_averages': daily_averages,
            'overall_trends': {
                'mood_trend': mood_trend,
                'stress_trend': stress_trend,
                'energy_trend': energy_trend,
                'productivity_trend': productivity_trend,
                'sentiment_trend': sentiment_trend
            }
        }
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate simple linear trend (positive = increasing, negative = decreasing)."""
        if len(values) < 2:
            return 0.0
        
        # Simple trend: compare first half average to second half average
        mid_point = len(values) // 2
        first_half_avg = sum(values[:mid_point]) / mid_point if mid_point > 0 else 0
        second_half_avg = sum(values[mid_point:]) / (len(values) - mid_point) if len(values) > mid_point else 0
        
        return second_half_avg - first_half_avg
    
    def get_sentiment_analysis(self, days: int = 30) -> Dict:
        """Get detailed sentiment analysis."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT sentiment_label, COUNT(*) as count, AVG(sentiment_score) as avg_score
            FROM journal_entries 
            WHERE timestamp > datetime('now', '-{} days')
            GROUP BY sentiment_label
        '''.format(days))
        
        rows = cursor.fetchall()
        conn.close()
        
        sentiment_distribution = {}
        total_entries = 0
        
        for row in rows:
            sentiment_distribution[row[0]] = {
                'count': row[1],
                'avg_score': row[2] or 0
            }
            total_entries += row[1]
        
        # Calculate percentages
        for sentiment in sentiment_distribution:
            sentiment_distribution[sentiment]['percentage'] = (
                sentiment_distribution[sentiment]['count'] / total_entries * 100
                if total_entries > 0 else 0
            )
        
        return {
            'period_days': days,
            'total_entries': total_entries,
            'sentiment_distribution': sentiment_distribution
        }
    
    def get_burnout_indicators(self, days: int = 30) -> Dict:
        """Calculate burnout risk indicators from journal data."""
        trends = self.get_mood_trends(days)
        sentiment = self.get_sentiment_analysis(days)
        
        # Burnout risk factors
        declining_mood = trends['overall_trends']['mood_trend'] < -0.1
        increasing_stress = trends['overall_trends']['stress_trend'] > 0.1
        declining_energy = trends['overall_trends']['energy_trend'] < -0.1
        declining_productivity = trends['overall_trends']['productivity_trend'] < -0.1
        negative_sentiment = sentiment['sentiment_distribution'].get('negative', {}).get('percentage', 0) > 40
        
        # Calculate burnout risk score (0-1)
        risk_factors = [
            declining_mood,
            increasing_stress,
            declining_energy,
            declining_productivity,
            negative_sentiment
        ]
        
        burnout_score = sum(risk_factors) / len(risk_factors)
        
        # Get recent keywords that might indicate stress
        recent_entries = self.get_recent_entries(7, 20)
        stress_keywords = []
        
        for entry in recent_entries:
            for keyword in entry['keywords']:
                if keyword in ['tired', 'exhausted', 'overwhelmed', 'stressed', 'burnout', 
                             'anxious', 'worried', 'pressure', 'deadline', 'difficult']:
                    stress_keywords.append(keyword)
        
        return {
            'burnout_risk_score': burnout_score,
            'risk_factors': {
                'declining_mood': declining_mood,
                'increasing_stress': increasing_stress,
                'declining_energy': declining_energy,
                'declining_productivity': declining_productivity,
                'negative_sentiment_dominance': negative_sentiment
            },
            'stress_keywords': stress_keywords[:10],  # Top 10 stress keywords
            'recommendation': self._get_burnout_recommendation(burnout_score)
        }
    
    def _get_burnout_recommendation(self, score: float) -> str:
        """Get recommendation based on burnout risk score."""
        if score < 0.2:
            return "Low burnout risk - Keep maintaining healthy habits"
        elif score < 0.4:
            return "Moderate burnout risk - Consider stress management techniques"
        elif score < 0.6:
            return "High burnout risk - Prioritize self-care and work-life balance"
        elif score < 0.8:
            return "Very high burnout risk - Take immediate action to reduce stress"
        else:
            return "Critical burnout risk - Seek professional support and significantly reduce workload"
    
    def create_daily_prompt(self) -> str:
        """Generate a daily journaling prompt."""
        prompts = [
            "How was your energy level today compared to yesterday?",
            "What was the most challenging part of your day?",
            "What accomplishment are you most proud of today?",
            "How did you handle stress today?",
            "What drained your energy the most today?",
            "What gave you the most energy today?",
            "How would you rate your focus today?",
            "What was your biggest distraction today?",
            "How balanced did your work and personal time feel today?",
            "What are you grateful for today?",
            "What would make tomorrow better?",
            "How did you feel about your productivity today?"
        ]
        
        import random
        return random.choice(prompts)


if __name__ == "__main__":
    import random
    
    journal = JournalingSystem()
    
    # Add sample entries
    sample_entries = [
        ("Today was productive but I'm feeling tired. Had many meetings but accomplished a lot.", 6, 7, 4, 7),
        ("Feeling overwhelmed with deadlines. Too much pressure and not enough time.", 3, 9, 2, 4),
        ("Great day! Finished the project early and had time to relax.", 8, 2, 8, 9),
        ("Struggled to focus today. Mind was wandering and felt unmotivated.", 4, 5, 3, 3)
    ]
    
    for text, mood, stress, energy, productivity in sample_entries:
        entry = journal.add_entry(text, mood, stress, energy, productivity)
        print(f"Added entry: {entry['sentiment_label']} sentiment")
    
    # Display analysis
    trends = journal.get_mood_trends(7)
    print("\nMood trends:", json.dumps(trends, indent=2, default=str))
    
    burnout = journal.get_burnout_indicators(7)
    print("\nBurnout indicators:", json.dumps(burnout, indent=2, default=str))
