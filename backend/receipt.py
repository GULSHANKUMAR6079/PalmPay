import os
from datetime import datetime

from reportlab.lib.pagesizes import A6
from reportlab.pdfgen import canvas


def generate_receipt(
    out_dir: str,
    transaction_id: int,
    customer_name: str,
    masked_upi: str,
    amount_rupees: float,
    merchant_id: str,
    razorpay_payment_id: str,
) -> str:
    """Small receipt-sized PDF a merchant can print at the counter."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"receipt_{transaction_id}.pdf")

    c = canvas.Canvas(path, pagesize=A6)
    width, height = A6
    y = height - 20 * 1.0

    def line(text, size=9, dy=14, bold=False):
        nonlocal y
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        c.drawString(15, y, text)
        y -= dy

    line("Palm Pay -- Payment Receipt", size=12, bold=True, dy=20)
    line(f"Transaction ID : {transaction_id}")
    line(f"Date/Time      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    line(f"Merchant ID    : {merchant_id}")
    line(f"Customer       : {customer_name}")
    line(f"UPI ID         : {masked_upi}")
    line(f"Amount Paid    : Rs {amount_rupees:.2f}")
    line(f"Payment Ref.   : {razorpay_payment_id}")
    line("")
    line("Thank you!", size=9)

    c.save()
    return path


def mask_vpa(vpa: str) -> str:
    """upi id shown on receipts/screen shouldn't be the full identifier."""
    if "@" not in vpa:
        return "****"
    user, bank = vpa.split("@", 1)
    if len(user) <= 2:
        masked_user = "*" * len(user)
    else:
        masked_user = user[0] + "*" * (len(user) - 2) + user[-1]
    return f"{masked_user}@{bank}"
