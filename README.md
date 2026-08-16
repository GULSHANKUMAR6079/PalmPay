# Palm Pay — palm-biometric micro-payments prototype

Two-scan palm payment flow: **identify → merchant enters amount → re-scan
palm to authorize → receipt**, charged through a pre-registered Razorpay
UPI Autopay mandate, capped at ₹100/transaction.

## Architecture

```text
palm image ──▶ MediaPipe HandLandmarker ──▶ align_palm() ──▶ HOG + PCA ──▶ embedding
 (webcam)         (detector.py)              (detector.py)   (embedder.py)     │
                                                                                 ▼
                                                                    PalmMatcher.identify()/verify()
                                                                                 │
                                                                                 ▼
                                                              FastAPI (main.py) ──▶ Razorpay UPI
                                                                    │              Autopay mandate
                                                                    ▼
                                                            SQLite (models.py) + PDF receipt
```

- **`backend/palm/detector.py`** — MediaPipe finds the hand (21 landmarks);
  `align_palm()` uses 4 stable landmarks (wrist, index/middle/pinky MCP) to
  crop/rotate/scale the palm into a consistent pose.
- **`backend/palm/embedder.py`** — HOG texture features + PCA → a 128-D
  identity vector. Classical, explainable, CPU-only. Swap for a fine-tuned
  CNN later if you want a stronger accuracy baseline to compare against.
- **`backend/palm/matcher.py`** — in-memory cosine-similarity search.
  `identify()` finds the best match across everyone; `verify()` checks the
  *second* scan matches the *same* customer as the first (stops a palm swap
  mid-transaction from authorizing the wrong person's payment).
- **`backend/payments/razorpay_client.py`** — wraps Razorpay's S2S UPI
  Autopay flow (mandate registration once, token-based charges after).
- **`backend/main.py`** — FastAPI endpoints tying it together.
- **`frontend/index.html`** — merchant terminal demo (webcam + fetch calls).

## Setup

```bash
pip install -r requirements.txt

# Download the MediaPipe hand landmark model (~12 MB)
curl -o hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

# Fit the embedder's PCA on a bootstrap batch of palm photos BEFORE first run.
# Put whatever photos you have in one folder -- even just 1 per hand per
# teammate works, the script tops it up with augmented variants and prints a
# summary of successful detections vs failures.
python scripts/fit_pca.py --photos_dir ./bootstrap_photos --out ./pca.joblib

# Razorpay TEST MODE keys — place them in a `.env` file in the project root:
# RAZORPAY_KEY_ID=rzp_test_xxxxxxxx
# RAZORPAY_KEY_SECRET=xxxxxxxxxxxxxxxx
# (python-dotenv automatically loads .env on startup so you don't need manual exports)

uvicorn backend.main:app --reload
# then open frontend/index.html in a browser (serve it, don't just double-click
# the file, or the camera permission prompt may not fire)
```

Run the offline self-test any time (no camera / model file / Razorpay needed):

```bash
python -m backend.palm.test_pipeline
```

Inspect registered customers and mandate status:

```bash
python scripts/list_customers.py
```

Debug endpoint: `GET /customers/{customer_id}` returns a JSON summary of the customer's mandate status and enrolled palm embeddings.

## Things to fix before this touches a real customer's real money

1. **The mandate-approval step is not fully wired up.** `create_authorisation_payment()`
   in `razorpay_client.py` needs the customer to approve the mandate inside
   their own UPI app (PIN entry) — that's a redirect/QR/intent-link flow on
   Razorpay's side, not something a headless backend call can do alone.
   `/webhooks/razorpay/mandate-approved` can be called to simulate approval,
   and contains webhook signature verification scaffolding (`X-Razorpay-Signature`
   verified against `RAZORPAY_WEBHOOK_SECRET` when configured).
2. **Stay in Razorpay test mode** until your guide/institution has reviewed
   consent and data handling — this system stores biometric data (palm
   embeddings) linked to a financial identifier (UPI ID), which is sensitive
   under India's DPDP Act, and it moves real money once switched to live keys.
3. **Don't store raw palm photos.** `main.py` already discards them after
   embedding — only the embedding vector is persisted. Keep it that way, and
   encrypt the database at rest.
4. **Tune `PalmMatcher`'s threshold empirically**, and report both False
   Accept Rate and False Reject Rate in your evaluation — a false accept
   means charging the wrong person, which is the failure mode that actually
   matters for a payment system.
5. Test everything on volunteers (classmates, staff) before any pilot with
   real underbanked customers, and get institutional sign-off first — that
   population needs extra care around informed consent.

## Dataset — where this stands right now vs. what a paper needs

Right now you likely have ~1 photo per hand per teammate. `fit_pca.py` and
`/customers/register` both auto-augment (`backend/palm/augment.py`) to get
the pipeline running end-to-end with that -- and the self-test above
confirms the plumbing works even at this scale. But augmentation only
jitters rotation/brightness/noise around a photo you already have; it
can't invent a genuinely different session, lighting, or hand condition.

For the evaluation section of an actual paper, you'll want a second,
**separate** round of real data collection before you compute FAR/FRR:
more people (ideally 20-30+, not just your team), several genuine photos
per hand, taken on more than one day. Report accuracy on that real data,
not on augmented copies of a handful of photos -- a reviewer will ask.

## Why this is a reasonable paper topic

Most published palm-biometric payment work targets enterprise/retail
(Amazon One and similar). A system built specifically for **underbanked
customers with no smartphone/card**, using an **RBI-compliant small-value
mandate architecture** instead of trying to bypass authentication, is a
fairly clean gap. Your strongest evaluation section: FAR/FRR of the
HOG+PCA baseline across N enrolled palms, effect of lighting/angle/hand-size
variation, and (if you build it) a comparison against a fine-tuned CNN
embedding as an ablation.
