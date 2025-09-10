from sqlalchemy import create_engine, Column, String, Text, DateTime
from sqlalchemy.orm import declarative_base, Session
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from .config import settings
import json

Base = declarative_base()

class Memory(Base):
    __tablename__ = "memories"
    user_id = Column(String, primary_key=True)
    key = Column(String, primary_key=True)
    value = Column(Text)  # store JSON string
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

engine = create_engine(settings.db_url, echo=False, future=True)
Base.metadata.create_all(engine)

def upsert_memory(user_id: str, key: str, value: Dict[str, Any]):
    with Session(engine) as sess:
        val_str = json.dumps(value, ensure_ascii=False)
        existing = sess.get(Memory, (user_id, key))
        if existing:
            existing.value = val_str
            existing.updated_at = datetime.now(timezone.utc)
        else:
            sess.add(Memory(user_id=user_id, key=key, value=val_str))
        sess.commit()

def query_memory(user_id: str, key: str) -> Optional[Dict[str, Any]]:
    with Session(engine) as sess:
        obj = sess.get(Memory, (user_id, key))
        if obj and obj.value:
            try:
                return json.loads(obj.value)
            except Exception:
                return None
        return None
