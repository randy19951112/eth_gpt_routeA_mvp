from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean
from sqlalchemy.orm import declarative_base, Session
from datetime import datetime, timezone
from typing import List, Optional
from .config import settings
import json

Base = declarative_base()

class Knowledge(Base):
    __tablename__ = "knowledge"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, index=True, nullable=False)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    tags = Column(String, default="")  # comma-separated
    active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

engine = create_engine(settings.db_url, echo=False, future=True)
Base.metadata.create_all(engine)

def upsert_knowledge(user_id: str, title: str, content: str, tags: Optional[list]=None, active: bool=True) -> int:
    with Session(engine) as sess:
        k = Knowledge(user_id=user_id, title=title, content=content, tags=",".join(tags or []), active=active)
        sess.add(k)
        sess.commit()
        return k.id

def search_knowledge(user_id: str, query: Optional[str]=None, tags: Optional[list]=None, limit: int=5) -> list:
    with Session(engine) as sess:
        q = sess.query(Knowledge).filter(Knowledge.user_id==user_id, Knowledge.active==True)
        if query:
            like = f"%{query}%"
            q = q.filter((Knowledge.title.ilike(like)) | (Knowledge.content.ilike(like)))
        if tags:
            for t in tags:
                like_t = f"%{t}%"
                q = q.filter(Knowledge.tags.ilike(like_t))
        q = q.order_by(Knowledge.created_at.desc()).limit(limit)
        rows = q.all()
        return [{"id": r.id, "title": r.title, "content": r.content, "tags": (r.tags or "").split(","), "created_at": r.created_at.isoformat()} for r in rows]
