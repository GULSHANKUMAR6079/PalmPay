from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class CustomerStateResponse(BaseModel):
    id: int
    name: str
    contact: str
    email: Optional[str] = None
    masked_upi: str
    razorpay_customer_id: Optional[str] = None
    mandate_order_id: Optional[str] = None
    mandate_token_id: Optional[str] = None
    mandate_approved: bool
    embedding_count: int
    created_at: datetime


class RegisterResponse(BaseModel):
    customer_id: int
    mandate_order_id: str
    message: str = "Mandate order created. Customer must approve it once in their UPI app."


class MandateApprovedRequest(BaseModel):
    """Called by your Razorpay webhook handler once the customer approves
    the mandate in their UPI app and a token_id is issued."""
    customer_id: int
    token_id: str


class IdentifyResponse(BaseModel):
    matched: bool
    customer_id: Optional[int] = None
    name: Optional[str] = None
    masked_upi: Optional[str] = None
    confidence: float
    session_id: Optional[int] = None  # == transaction id, used for the next steps


class SetAmountRequest(BaseModel):
    session_id: int
    amount_rupees: float


class AuthorizeResponse(BaseModel):
    status: str
    razorpay_payment_id: Optional[str] = None
    receipt_url: Optional[str] = None
    reason: Optional[str] = None
