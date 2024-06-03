import datetime
from typing import Optional

from sqlalchemy import BigInteger, Date, Float, Integer
from schemas.base import Base
from sqlalchemy.orm import Mapped, mapped_column


class DataByWeek(Base):
    __tablename__='data_by_week'

    product_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    category_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    date: Mapped[datetime.date] = mapped_column(Date, primary_key=True)
    max_price: Mapped[float] = mapped_column(Float(8))
    min_price: Mapped[float] = mapped_column(Float(8))
    avg_price: Mapped[float] = mapped_column(Float(8))
    view: Mapped[int] = mapped_column(Integer())
    cart: Mapped[int] = mapped_column(Integer())
    remove_from_cart: Mapped[int] = mapped_column(Integer())
    purchase: Mapped[int] = mapped_column(Integer())
    rank: Mapped[Optional[int]] = mapped_column(Integer(), nullable=True)
    rank_in_category: Mapped[Optional[int]] = mapped_column(Integer(), nullable=True)
    