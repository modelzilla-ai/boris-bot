"""
memory.py -- Memória persistente com SQLite
============================================
Armazena histórico de preços, notícias e decisões em banco SQLite.
"""

import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

DB_FILE = Path("boris_memory.db")
MAX_PRICE_HISTORY = 48
MAX_NEWS_HISTORY = 30
MAX_DECISION_HISTORY = 20

class AgentMemory:
    def __init__(self, db_file: Path = DB_FILE):
        self.db_file = db_file
        self._conn = None
        self._init_db()

    def _connect(self):
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_file, detect_types=sqlite3.PARSE_DECLTYPES)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_db(self):
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS price_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                price_usd REAL NOT NULL,
                change_24h REAL NOT NULL,
                volume_24h REAL NOT NULL,
                timestamp TEXT NOT NULL UNIQUE
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS news_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                source TEXT,
                summary TEXT,
                sentiment REAL,
                timestamp TEXT NOT NULL UNIQUE
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS decision_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trend TEXT NOT NULL,
                recommendation TEXT NOT NULL,
                timestamp TEXT NOT NULL UNIQUE
            )
        """)
        conn.commit()

    def add_price(self, price_data: dict):
        conn = self._connect()
        cur = conn.cursor()
        try:
            cur.execute("""
                INSERT INTO price_history (price_usd, change_24h, volume_24h, timestamp)
                VALUES (?, ?, ?, ?)
            """, (price_data['price_usd'], price_data['change_24h'], price_data['volume_24h'], price_data['timestamp']))
            conn.commit()
        except sqlite3.IntegrityError:
            logger.debug("Preço já existe na memória (timestamp duplicado).")
        # Manter apenas últimos MAX_PRICE_HISTORY registros
        cur.execute("""
            DELETE FROM price_history WHERE id NOT IN (
                SELECT id FROM price_history ORDER BY timestamp DESC LIMIT ?
            )
        """, (MAX_PRICE_HISTORY,))
        conn.commit()
        logger.debug("Preço adicionado: $%.2f", price_data['price_usd'])

    def get_recent_prices(self, n: int = 8) -> List[dict]:
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("""
            SELECT price_usd, change_24h, volume_24h, timestamp
            FROM price_history ORDER BY timestamp DESC LIMIT ?
        """, (n,))
        rows = cur.fetchall()
        return [dict(row) for row in rows[::-1]]  # ordem crescente

    def get_price_trend_summary(self) -> str:
        rows = self.get_recent_prices(8)
        if len(rows) < 2:
            return "Histórico insuficiente para calcular tendência."
        latest = rows[-1]
        earliest = rows[0]
        change = ((latest['price_usd'] - earliest['price_usd']) / earliest['price_usd']) * 100
        direction = "subiu" if change >= 0 else "caiu"
        return (f"Nas últimas ~24h o Bitcoin {direction} {abs(change):.2f}% "
                f"(de ${earliest['price_usd']:,.2f} para ${latest['price_usd']:,.2f}).")

    def get_last_price(self) -> Optional[dict]:
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("SELECT * FROM price_history ORDER BY timestamp DESC LIMIT 1")
        row = cur.fetchone()
        return dict(row) if row else None

    def add_news(self, news_list: List[dict]):
        conn = self._connect()
        cur = conn.cursor()
        for n in news_list:
            try:
                cur.execute("""
                    INSERT INTO news_history (title, source, summary, sentiment, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (n['title'], n.get('source'), n.get('summary'), n.get('sentiment'),
                      datetime.utcnow().isoformat() + "Z"))
                conn.commit()
            except sqlite3.IntegrityError:
                logger.debug("Notícia duplicada ignorada: %s", n['title'][:50])
        # Manter últimos MAX_NEWS_HISTORY registros
        cur.execute("""
            DELETE FROM news_history WHERE id NOT IN (
                SELECT id FROM news_history ORDER BY timestamp DESC LIMIT ?
            )
        """, (MAX_NEWS_HISTORY,))
        conn.commit()

    def get_recent_news(self, n: int = 3) -> List[dict]:
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("""
            SELECT title, source, summary, sentiment, timestamp
            FROM news_history ORDER BY timestamp DESC LIMIT ?
        """, (n,))
        rows = cur.fetchall()
        return [dict(row) for row in rows]

    def add_decision(self, trend: str, recommendation: str):
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO decision_history (trend, recommendation, timestamp)
            VALUES (?, ?, ?)
        """, (trend, recommendation, datetime.utcnow().isoformat() + "Z"))
        conn.commit()
        # Manter últimos MAX_DECISION_HISTORY
        cur.execute("""
            DELETE FROM decision_history WHERE id NOT IN (
                SELECT id FROM decision_history ORDER BY timestamp DESC LIMIT ?
            )
        """, (MAX_DECISION_HISTORY,))
        conn.commit()

    def get_last_decision(self) -> Optional[dict]:
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("SELECT * FROM decision_history ORDER BY timestamp DESC LIMIT 1")
        row = cur.fetchone()
        return dict(row) if row else None

    def get_decision_summary(self) -> str:
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("SELECT trend, recommendation, timestamp FROM decision_history ORDER BY timestamp DESC LIMIT 5")
        rows = cur.fetchall()
        if not rows:
            return "Nenhuma decisão anterior registrada."
        lines = ["Últimas decisões do agente:"]
        for row in rows:
            lines.append(f"  - [{row['timestamp'][:10]}] {row['trend']}: {row['recommendation']}")
        return "\n".join(lines)

    def should_send_alert(self, current_change: float, threshold: float) -> bool:
        if abs(current_change) < threshold:
            return False
        last = self.get_last_decision()
        if not last:
            return True
        try:
            last_time = datetime.fromisoformat(last['timestamp'].replace('Z', '+00:00'))
            now = datetime.utcnow().replace(tzinfo=last_time.tzinfo)
            return (now - last_time) > timedelta(minutes=30)
        except Exception:
            return True

    def clear(self):
        conn = self._connect()
        conn.execute("DELETE FROM price_history")
        conn.execute("DELETE FROM news_history")
        conn.execute("DELETE FROM decision_history")
        conn.commit()
        logger.warning("Memória limpa completamente.")

    def stats(self) -> dict:
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM price_history")
        price_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM news_history")
        news_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM decision_history")
        decision_count = cur.fetchone()[0]
        return {
            "price_records": price_count,
            "news_records": news_count,
            "decision_records": decision_count,
            "memory_file": str(self.db_file.absolute()),
        }