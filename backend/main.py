"""
Palm Pay -- prototype backend.

Flow implemented here (matches the two-scan UX you described):

  1. POST /session/identify   -- customer shows palm -> identify them
  2. POST /session/set-amount -- merchant enters amount, customer sees it
  3. POST /session/authorize  -- customer shows palm AGAIN to confirm
                                  -> re-verify it's the SAME person as
                                     step 1, then charge via the customer's
                                     pre-registered UPI Autopay mandate
  4. GET  /receipts/{id}      -- printable PDF receipt

Registration (customer + mandate setup) is a separate, one-time flow --
see /customers/register and the note in README.md about why mandate
approval can't be fully headless.

Run with (after `pip install -r requirements.txt` and downloading the
MediaPipe model -- see README.md):
    uvicorn backend.main:app --reload
"""

import io
import os
import re
from typing import List, Optional

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.orm import Session

from backend.database import Base, engine, get_db
from backend.models import Customer, PalmEmbedding, Transaction, TransactionStatus
from backend.palm.augment import augment_palm_image
from backend.palm.detector import PalmDetector, align_palm
from backend.palm.embedder import PalmEmbedder
from backend.palm.matcher import PalmMatcher
from backend.payments.razorpay_client import RazorpayMandateClient
from backend.receipt import generate_receipt, mask_vpa
from backend.schemas import (
    AuthorizeResponse, CustomerStateResponse, IdentifyResponse, MandateApprovedRequest,
    RegisterResponse, SetAmountRequest,
)
from dotenv import load_dotenv
load_dotenv()

# Regex patterns for input validation
EMAIL_REGEX = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
PHONE_REGEX = r"^\+?[0-9]{7,15}$"
VPA_REGEX = r"^[\w.-]+@[\w.-]+$"

# --- hard business rule, enforced server-side regardless of what the client sends ---
PAYMENT_CAP_RUPEES = 100.0

# How many template embeddings we want stored per customer. If fewer real
# photos are uploaded than this, we top up with augmented variants of the
# real ones so matching isn't resting on a single static image -- see
# backend/palm/augment.py for why that's a stopgap, not a fix for having
# too few real photos overall.
TEMPLATES_PER_CUSTOMER = 6
MIN_REAL_PHOTOS = 1

RECEIPTS_DIR = os.environ.get("RECEIPTS_DIR", "./receipts")
MODEL_PATH = os.environ.get("HAND_LANDMARKER_MODEL", "hand_landmarker.task")
PCA_PATH = os.environ.get("PALM_PCA_PATH", "./pca.joblib")

Base.metadata.create_all(bind=engine)
app = FastAPI(title="Palm Pay Prototype")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler returning structured JSON responses for all unhandled errors."""
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=exc.headers,
        )
    message = str(exc) if str(exc) else f"Internal Server Error ({exc.__class__.__name__})"
    return JSONResponse(
        status_code=500,
        content={"detail": message},
    )


MATCH_THRESHOLD = float(os.environ.get("MATCH_THRESHOLD", "0.70"))
AUTO_APPROVE_MANDATE = os.environ.get("AUTO_APPROVE_MANDATE", "true").lower() == "true"

detector = PalmDetector(model_path=MODEL_PATH)
# embedding_dim here only matters if load() below finds nothing and a
# fresh embedder is fit later -- the loaded PCA (fit via scripts/fit_pca.py)
# determines the real output dimension. Keep this in sync with that script.
embedder = PalmEmbedder(embedding_dim=24)
if os.path.exists(PCA_PATH):
    embedder.load(PCA_PATH)
matcher = PalmMatcher(match_threshold=MATCH_THRESHOLD)

razorpay_client = RazorpayMandateClient()  # reads RAZORPAY_KEY_ID / SECRET from env


@app.on_event("startup")
def load_existing_embeddings():
    """Matcher is in-memory, so repopulate it from the DB on every restart."""
    db = next(get_db())
    for emb_row in db.query(PalmEmbedding).all():
        matcher.add(customer_id=emb_row.customer_id, embedding=np.array(emb_row.vector))


def _read_upload_as_bgr(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode uploaded image")
    return img


def _detect_align_embed(file_bytes: bytes) -> np.ndarray:
    frame = _read_upload_as_bgr(file_bytes)
    landmarks = detector.detect(frame)
    if landmarks is None:
        raise HTTPException(422, "No hand detected in image")
    aligned = align_palm(frame, landmarks)
    if aligned is None:
        raise HTTPException(422, "Could not align palm (hand too small/ambiguous pose)")
    return embedder.embed(aligned)


# ---------------------------------------------------------------------------
# Registration (one-time per customer)
# ---------------------------------------------------------------------------

@app.post("/customers/register", response_model=RegisterResponse)
def register_customer(
    name: str = Form(...),
    contact: str = Form(...),
    email: str = Form(...),
    upi_vpa: str = Form(...),
    palm_photos: List[UploadFile] = File(
        ..., description=f"At least {MIN_REAL_PHOTOS} palm photo. More real photos (different "
                          f"angles/sessions) is always better than relying on augmentation to top up."
    ),
    db: Session = Depends(get_db),
):
    # Field input validation
    name_clean = name.strip()
    contact_clean = contact.strip()
    email_clean = email.strip().lower()
    upi_vpa_clean = upi_vpa.strip()

    if not name_clean:
        raise HTTPException(400, "Name is required")
    if not re.match(EMAIL_REGEX, email_clean):
        raise HTTPException(400, "Invalid email address format")
    if not re.match(PHONE_REGEX, contact_clean):
        raise HTTPException(400, "Invalid phone number format (expected 7 to 15 digits)")
    if not re.match(VPA_REGEX, upi_vpa_clean):
        raise HTTPException(400, "Invalid UPI ID format (e.g. name@bank)")

    # Deduplication check against local database
    existing = db.query(Customer).filter(
        (Customer.email == email_clean) | (Customer.contact == contact_clean)
    ).first()
    if existing:
        if existing.email == email_clean:
            raise HTTPException(400, f"A customer with email '{email_clean}' already exists")
        else:
            raise HTTPException(400, f"A customer with phone number '{contact_clean}' already exists")

    if len(palm_photos) < MIN_REAL_PHOTOS:
        raise HTTPException(400, f"Please provide at least {MIN_REAL_PHOTOS} palm photo(s)")

    aligned_real = []
    for photo in palm_photos:
        frame = _read_upload_as_bgr(photo.file.read())
        landmarks = detector.detect(frame)
        if landmarks is None:
            raise HTTPException(422, "No hand detected in one of the uploaded photos")
        aligned = align_palm(frame, landmarks)
        if aligned is None:
            raise HTTPException(422, "Could not align palm in one of the uploaded photos")
        aligned_real.append(aligned)

    # Top up with augmented variants if we got fewer real photos than
    # TEMPLATES_PER_CUSTOMER, so the matcher has more than one static
    # reference point per person. Real photos are always used as-is too.
    templates = list(aligned_real)
    if len(templates) < TEMPLATES_PER_CUSTOMER:
        needed = TEMPLATES_PER_CUSTOMER - len(templates)
        per_photo = -(-needed // len(aligned_real))  # ceil division
        for i, img in enumerate(aligned_real):
            templates.extend(augment_palm_image(img, n_variants=per_photo, seed=i))
        templates = templates[:max(TEMPLATES_PER_CUSTOMER, len(aligned_real))]

    embeddings = [embedder.embed(img) for img in templates]

    # End-to-end transactional execution
    try:
        customer = Customer(
            name=name_clean,
            contact=contact_clean,
            email=email_clean,
            upi_vpa=upi_vpa_clean
        )
        db.add(customer)
        db.flush()  # assigns customer.id without committing yet

        # Razorpay side: create customer + mandate order
        rp_customer_id = razorpay_client.create_customer(
            name=name_clean,
            contact=contact_clean,
            email=email_clean
        )
        mandate_order_id = razorpay_client.create_mandate_order(
            razorpay_customer_id=rp_customer_id,
            mandate_limit_paise=int(PAYMENT_CAP_RUPEES * 100),
        )
        customer.razorpay_customer_id = rp_customer_id
        customer.mandate_order_id = mandate_order_id
        customer.mandate_limit_paise = int(PAYMENT_CAP_RUPEES * 100)
        if AUTO_APPROVE_MANDATE:
            customer.mandate_token_id = f"mock_token_{customer.id}"

        for vec in embeddings:
            db.add(PalmEmbedding(customer_id=customer.id, vector=vec.tolist()))

        db.commit()
        db.refresh(customer)

        # Update in-memory matcher only AFTER DB commit succeeds
        for vec in embeddings:
            matcher.add(customer_id=customer.id, embedding=vec)

        return RegisterResponse(customer_id=customer.id, mandate_order_id=mandate_order_id)
    except Exception as e:
        db.rollback()
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(500, f"Registration failed during payment gateway setup: {str(e)}")


@app.get("/customers/{customer_id}", response_model=CustomerStateResponse)
def get_customer_state(customer_id: int, db: Session = Depends(get_db)):
    """Debug endpoint to inspect customer registration & mandate status."""
    customer = db.query(Customer).get(customer_id)
    if not customer:
        raise HTTPException(404, f"Customer with ID {customer_id} not found")

    return CustomerStateResponse(
        id=customer.id,
        name=customer.name,
        contact=customer.contact,
        email=customer.email,
        masked_upi=mask_vpa(customer.upi_vpa),
        razorpay_customer_id=customer.razorpay_customer_id,
        mandate_order_id=customer.mandate_order_id,
        mandate_token_id=customer.mandate_token_id,
        mandate_approved=customer.mandate_token_id is not None,
        embedding_count=len(customer.embeddings),
        created_at=customer.created_at,
    )


@app.post("/webhooks/razorpay/mandate-approved")
async def mandate_approved(
    req: MandateApprovedRequest,
    request: Request,
    db: Session = Depends(get_db),
    x_razorpay_signature: Optional[str] = Header(None, alias="X-Razorpay-Signature"),
):
    """In production this is a signed Razorpay webhook callback.
    If RAZORPAY_WEBHOOK_SECRET is set in environment, signature verification is enforced.
    """
    webhook_secret = os.environ.get("RAZORPAY_WEBHOOK_SECRET")
    if webhook_secret:
        if not x_razorpay_signature:
            raise HTTPException(400, "Missing X-Razorpay-Signature header")
        raw_body = await request.body()
        is_valid = razorpay_client.verify_webhook_signature(
            body_bytes=raw_body,
            signature=x_razorpay_signature,
            webhook_secret=webhook_secret,
        )
        if not is_valid:
            raise HTTPException(400, "Invalid Razorpay webhook signature")

    customer = db.query(Customer).get(req.customer_id)
    if not customer:
        raise HTTPException(404, "Unknown customer")
    customer.mandate_token_id = req.token_id
    db.commit()
    return {"ok": True, "mandate_approved": True}


# ---------------------------------------------------------------------------
# Payment flow (the two-scan UX)
# ---------------------------------------------------------------------------

@app.post("/session/identify", response_model=IdentifyResponse)
def identify(
    merchant_id: str = Form(...),
    palm_photo: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    embedding = _detect_align_embed(palm_photo.file.read())
    customer_id, confidence = matcher.identify(embedding)

    if customer_id is None:
        return IdentifyResponse(matched=False, confidence=confidence)

    customer = db.query(Customer).get(customer_id)
    if customer.mandate_token_id is None:
        raise HTTPException(409, "Customer identified but has not completed mandate approval yet")

    txn = Transaction(
        customer_id=customer.id,
        merchant_id=merchant_id,
        status=TransactionStatus.IDENTIFIED,
        identify_confidence=confidence,
    )
    db.add(txn)
    db.commit()
    db.refresh(txn)

    return IdentifyResponse(
        matched=True,
        customer_id=customer.id,
        name=customer.name,
        masked_upi=mask_vpa(customer.upi_vpa),
        confidence=confidence,
        session_id=txn.id,
    )


@app.post("/session/set-amount")
def set_amount(req: SetAmountRequest, db: Session = Depends(get_db)):
    txn = db.query(Transaction).get(req.session_id)
    if not txn or txn.status != TransactionStatus.IDENTIFIED:
        raise HTTPException(409, "Session not in a state that accepts an amount")

    if req.amount_rupees <= 0 or req.amount_rupees > PAYMENT_CAP_RUPEES:
        raise HTTPException(400, f"Amount must be between Rs 0 and Rs {PAYMENT_CAP_RUPEES}")

    txn.amount_rupees = req.amount_rupees
    txn.status = TransactionStatus.AMOUNT_SET
    db.commit()
    return {"ok": True, "amount_rupees": req.amount_rupees}


@app.post("/session/authorize", response_model=AuthorizeResponse)
def authorize(
    session_id: int = Form(...),
    palm_photo: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    txn = db.query(Transaction).get(session_id)
    if not txn or txn.status != TransactionStatus.AMOUNT_SET:
        raise HTTPException(409, "Session not ready for authorization")

    embedding = _detect_align_embed(palm_photo.file.read())
    same_person, score = matcher.verify(customer_id=txn.customer_id, embedding=embedding)
    txn.authorize_confidence = score

    if not same_person:
        txn.status = TransactionStatus.REJECTED_MISMATCH
        db.commit()
        return AuthorizeResponse(status="rejected_mismatch", reason="Palm did not match the identified customer")

    customer = db.query(Customer).get(txn.customer_id)
    try:
        payment = razorpay_client.charge_with_token(
            token_id=customer.mandate_token_id,
            razorpay_customer_id=customer.razorpay_customer_id,
            amount_rupees=txn.amount_rupees,
            mandate_limit_paise=customer.mandate_limit_paise,
        )
    except Exception as e:  # noqa: BLE001 -- surface payment failures to the merchant UI
        txn.status = TransactionStatus.FAILED
        db.commit()
        return AuthorizeResponse(status="failed", reason=str(e))

    txn.status = TransactionStatus.PAID
    txn.razorpay_payment_id = payment["id"]

    receipt_path = generate_receipt(
        out_dir=RECEIPTS_DIR,
        transaction_id=txn.id,
        customer_name=customer.name,
        masked_upi=mask_vpa(customer.upi_vpa),
        amount_rupees=txn.amount_rupees,
        merchant_id=txn.merchant_id,
        razorpay_payment_id=payment["id"],
    )
    txn.receipt_path = receipt_path
    db.commit()

    return AuthorizeResponse(
        status="paid",
        razorpay_payment_id=payment["id"],
        receipt_url=f"/receipts/{txn.id}",
    )


@app.get("/receipts/{transaction_id}")
def get_receipt(transaction_id: int, db: Session = Depends(get_db)):
    txn = db.query(Transaction).get(transaction_id)
    if not txn or not txn.receipt_path:
        raise HTTPException(404, "Receipt not found")
    return FileResponse(txn.receipt_path, media_type="application/pdf")
