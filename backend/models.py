import enum
from datetime import datetime

from sqlalchemy import (
    JSON, Column, DateTime, Enum, Float, ForeignKey, Integer, String,
)
from sqlalchemy.orm import relationship

from backend.database import Base


class Customer(Base):
    __tablename__ = "customers"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    contact = Column(String, nullable=False, unique=True)
    email = Column(String, nullable=True, unique=True)

    # NOTE: store only what you need. `upi_vpa` is the customer's UPI ID,
    # not raw bank credentials -- Razorpay/NPCI hold the actual banking
    # relationship. Consider field-level encryption for this column.
    upi_vpa = Column(String, nullable=False)

    razorpay_customer_id = Column(String, nullable=True)
    mandate_order_id = Column(String, nullable=True)
    mandate_token_id = Column(String, nullable=True)   # set once mandate is approved
    mandate_limit_paise = Column(Integer, default=10000)  # Rs 100 default cap

    created_at = Column(DateTime, default=datetime.utcnow)

    embeddings = relationship("PalmEmbedding", back_populates="customer")
    transactions = relationship("Transaction", back_populates="customer")


class PalmEmbedding(Base):
    __tablename__ = "palm_embeddings"

    id = Column(Integer, primary_key=True)
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=False)

    # Embedding vector stored as a JSON list of floats. Fine at prototype
    # scale; move to a proper vector column/index if this grows large.
    vector = Column(JSON, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)

    customer = relationship("Customer", back_populates="embeddings")


class TransactionStatus(str, enum.Enum):
    IDENTIFIED = "identified"       # step 1 palm scan matched a customer
    AMOUNT_SET = "amount_set"       # step 2 merchant entered an amount
    AUTHORIZED = "authorized"       # step 3 second palm scan confirmed same person
    PAID = "paid"                   # Razorpay charge succeeded
    FAILED = "failed"               # Razorpay charge failed
    REJECTED_MISMATCH = "rejected_mismatch"  # second scan didn't match step 1 customer


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True)
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=False)
    merchant_id = Column(String, nullable=False)

    amount_rupees = Column(Float, nullable=True)
    status = Column(Enum(TransactionStatus), default=TransactionStatus.IDENTIFIED)

    identify_confidence = Column(Float, nullable=True)
    authorize_confidence = Column(Float, nullable=True)

    identify_handedness = Column(String, nullable=True)
    identify_embedding = Column(JSON, nullable=True)

    razorpay_payment_id = Column(String, nullable=True)
    receipt_path = Column(String, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    customer = relationship("Customer", back_populates="transactions")
