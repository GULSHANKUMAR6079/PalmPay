# End-to-End Hardware Assembly & Integration Guide for Palm Pay

> **Project:** Palm Pay — Infrared Palm Vein Biometric System  
> **Target Device:** Raspberry Pi 4B (4GB) with NoIR Camera & 850nm IR Array  
> **Skill Level:** Complete Beginner Friendly  

---

## 1. System Architecture & Component Purpose

Before connecting any wire, here is how each component in your list works together:

| Component | Purpose in Palm Pay |
| :--- | :--- |
| **Raspberry Pi 4B (4GB)** | Main computer. Runs Python backend, MediaPipe landmarking, PCA feature extraction, and payment API. |
| **NoIR Camera Module V2/V3** | Camera **without** an Infrared (IR) cut filter. It can see 850nm NIR light. |
| **850nm IR LEDs (8 pcs)** | Emits 850nm Near-Infrared light. Blood (hemoglobin) absorbs this wavelength, causing veins to appear dark. |
| **IR Bandpass Filter** | Mounted over the camera lens. Blocks visible light (room/sunlight) and allows **only** 850nm light to reach sensor. |
| **Current Resistors (220Ω–330Ω)** | Prevents LEDs from burning out by limiting electrical current from the 5V rail. |
| **Diode (1N4001 / 1N4148)** | Protection diode to prevent reverse-voltage damage to your circuit. |
| **Breadboard & Jumpers** | For solderless prototyping and testing before permanent perfboard assembly. |
| **Digital Multimeter** | Used to test voltage, current, and verify no short circuits before powering the Pi. |

---

## 2. Pinout & Polarities Reference (Crucial Safety Rules)

> [!CAUTION]
> **SAFETY RULE #1:** ALWAYS turn off and unplug the Raspberry Pi power supply before adding, removing, or changing any wire on the breadboard or GPIO header.

### 2.1 Component Polarities
1. **LEDs (850nm IR LEDs):**
   - **Anode (+):** Longer leg. Connects to 5V (via Resistor).
   - **Cathode (-):** Shorter leg (or side with a flat notch on plastic rim). Connects to Ground (GND).
2. **Diodes (1N4001 / 1N4148):**
   - **Anode (+):** Side WITHOUT line/stripe.
   - **Cathode (-):** Side WITH silver or black line/stripe. Current flows Anode → Cathode.
3. **Resistors (220Ω–330Ω):**
   - Non-directional (either orientation works). Color bands for 220Ω = Red-Red-Brown-Gold. 330Ω = Orange-Orange-Brown-Gold.

### 2.2 Raspberry Pi 4B GPIO Header Pins
Looking at the 40-pin header with the Pi oriented so the pins are at the top right:
- **Pin 2 (Top-Right outer pin):** 5V DC Power
- **Pin 4 (2nd outer pin):** 5V DC Power
- **Pin 6 (3rd outer pin):** Ground (GND)

```text
Raspberry Pi 4B Header (Top View, USB ports pointing down):
 [ 2 ]  5V Power   <--- Connect to Breadboard + Rail (via Diode)
 [ 4 ]  5V Power   <--- Connect to Fan (+)
 [ 6 ]  GND        <--- Connect to Breadboard - Rail & Fan (-)
```

---

## 3. Circuit Wiring Instructions (Breadboard Prototype)

### Why 5V Rail Instead of GPIO Pins?
A single Pi GPIO pin can only deliver ~16mA safely. 8 LEDs drawing ~15mA each need a total of **~120mA**, which will burn out a GPIO pin if driven directly. Therefore, we power the LEDs from the **5V Power Rail (Pin 2)** which is fed directly by your 5V/3A official power supply.

### Step-by-Step Circuit Assembly
1. **Power Rails:**
   - Connect a **Male-to-Female (M2F)** jumper wire from **Raspberry Pi Pin 2 (5V)** to the **Red (+) Rail** of your breadboard.
   - Connect an **M2F** jumper wire from **Raspberry Pi Pin 6 (GND)** to the **Blue (-) Rail** of your breadboard.
2. **Reverse Protection Diode:**
   - Insert Diode **1N4001** Anode (no stripe) into Red (+) Rail.
   - Insert Diode Cathode (with stripe) into an empty row (let's call it `Row A`).
3. **Parallel LED Array Wiring (For all 8 LEDs):**
   - Create 8 identical branches in parallel across the breadboard:
     - **Branch N:**
       - Take one **220Ω (or 330Ω) Resistor**: One leg in `Row A` (+), other leg in `Row B_N`.
       - Take one **850nm IR LED**: Long leg (Anode) in `Row B_N`, short leg (Cathode) into the **Blue (-) GND Rail**.
   - Repeat for all 8 LEDs.
4. **Physical Placement:**
   - Arrange the 8 LEDs on the breadboard (or perfboard) in a circular pattern surrounding the camera lens location so the light evenly illuminates the palm.

---

## 4. Pre-Power Safety & Multimeter Checks

> [!IMPORTANT]
> Perform these checks WITH THE PI UNPLUGGED FROM THE WALL POWER to prevent accidental short circuits!

1. **Short Circuit Check:**
   - Set multimeter to **Continuity Mode (Beep symbol)** or **Resistance (200Ω)**.
   - Touch Red probe to Breadboard Red (+) rail and Black probe to Blue (-) GND rail.
   - **Expected Result:** NO beep (or high resistance > 1000Ω). If it beeps or shows 0Ω, you have a short circuit—do NOT turn on power until fixed.
2. **LED Test:**
   - Set multimeter to **Diode Mode**.
   - Touch Red probe to LED Anode (+) and Black probe to LED Cathode (-).
   - **Note:** IR light (850nm) is invisible to human eyes! To verify it is on, open your smartphone front camera and look at the LED. You will see a faint purple/pink glow on the smartphone screen.

---

## 5. NoIR Camera & Optical Filter Setup

1. **Connecting NoIR Camera to Raspberry Pi 4B:**
   - Locate the **CSI Camera Connector** on the Pi 4B (between Micro-HDMI 2 and 3.5mm jack).
   - Gently pull up on the dark plastic locking clip.
   - Insert the ribbon cable with the **blue tape side facing the USB/Ethernet ports** (metallic pads facing the HDMI ports).
   - Push the locking clip back down securely.
2. **Installing the IR Bandpass Filter:**
   - Place the **850nm IR Bandpass Filter** directly over the camera lens.
   - Use small spots of hot glue or electrical tape around the outer edge of the filter housing (do NOT get glue on the glass optics!).
   - This filter blocks room lighting and sunlight, passing only the 850nm reflected light from palm veins.

---

## 6. Software Setup on Raspberry Pi OS

1. **Booting & Enabling Camera:**
   - Power on the Pi. Open terminal and update system:
     ```bash
     sudo apt update && sudo apt upgrade -y
     ```
   - Test camera feed:
     ```bash
     rpicam-hello
     # or on legacy OS: v4l2-ctl --list-devices
     ```
2. **Running the Palm Pay Codebase:**
   - Navigate to the repository:
     ```bash
     cd ~/palm-pay  # or path to repo
     pip install -r requirements.txt
     ```
   - Run the hardware & algorithm self-test:
     ```bash
     python -m backend.palm.test_pipeline
     ```
   - Launch the merchant server:
     ```bash
     uvicorn backend.main:app --host 0.0.0.0 --port 8000
     ```
   - Open browser at `http://localhost:8000` (or `frontend/index.html`) to test live palm identification!

---

## 7. Troubleshooting & Common Issues

| Problem | Cause | Solution |
| :--- | :--- | :--- |
| **LEDs not lighting up** | Reversed polarity or bad GND connection | Check LED long leg is to resistor/5V, short leg to GND. Use phone camera to see 850nm glow. |
| **Camera image is dark/black** | Bandpass filter attached without IR LEDs turned on | Ensure 5V supply is plugged in and all 8 IR LEDs are emitting 850nm light. |
| **No hand detected by MediaPipe** | Hand too close or poor contrast | Position hand 15–20cm above camera. Adjust LED angle so light is evenly spread. |
| **Pi overheating/throttling** | Heavy CPU load from MediaPipe / PCA | Attach heatsinks and ensure case fan is running (5V Pin 4 + GND Pin 6). |
