import datetime

from schemas.base import Base
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import BigInteger, Date, Float, Integer


class Product(Base):    
    __tablename__ = 'products'
    
    product_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    category_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    release_date: Mapped[datetime.date] = mapped_column(Date)
    max_price: Mapped[float] = mapped_column(Float(8))
    events_count: Mapped[int] = mapped_column(Integer)