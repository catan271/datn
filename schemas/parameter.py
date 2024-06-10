from schemas.base import Base
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import Float, Text


class Parameter(Base):    
    __tablename__ = 'parameters'
    
    name: Mapped[str] = mapped_column(Text, primary_key=True)
    value: Mapped[int] = mapped_column(Float(8), primary_key=True)