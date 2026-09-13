import io
import time
import warnings
import cv2
import easyocr
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

warnings.filterwarnings("ignore")

reader = easyocr.Reader(['en'], gpu=False)
OCR_ALLOWLIST = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

EXPECTED_CAPTCHA_LEN = 6

import os
DEBUG_DIR = "captcha_debug"
os.makedirs(DEBUG_DIR, exist_ok=True)


def preprocess_captcha(image_bytes):
    """
    Color-agnostic CAPTCHA cleaner — shape-aware, tuned to preserve letter
    shapes (v1 with median-blur(5) + ellipse-open(3,3) was cleaning dot
    noise well but rounding/smoothing letter corners enough to cause
    character-level misreads: L->E, v->Y, 6->b/e, U->u case flips.
    Lighter touch here keeps noise removal but preserves glyph shape.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, sat = cv2.threshold(hsv[:, :, 1], 35, 255, cv2.THRESH_BINARY)


    sat = cv2.medianBlur(sat, 3)


    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opened = cv2.morphologyEx(sat, cv2.MORPH_OPEN, open_kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    clean_mask = np.zeros_like(opened)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        if area > 60 and height > 10:
            clean_mask[labels == i] = 255

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, close_kernel)
    final_img = cv2.bitwise_not(cleaned)

    _, encoded_img = cv2.imencode('.png', final_img)
    return encoded_img.tobytes()


def verify_login_success(page, timeout_ms=8000):
    """
    Reliable login verification.

    Key fix vs old version:
    - Old code checked for "EXAM RESULTS" / "Log Out" text ANYWHERE on the page,
      even while still sitting on index.php. On this portal those labels are
      part of the static nav and are visible whether or not you're logged in,
      so a WRONG captcha was being reported as SUCCESS.
    - New logic: only trust text/frame markers once the URL has actually left
      index.php. If we're still on index.php with the 5 login boxes back,
      it's a guaranteed FAIL (wrong captcha/creds reloaded the form).
    """

    try:
        page.wait_for_load_state("networkidle", timeout=timeout_ms)
    except Exception:
        time.sleep(3)

    current_url = page.url.lower()

    if "students_corner" in current_url:
        return True


    try:
        still_on_login_form = page.locator("input[type='text']").count() >= 5
    except Exception:
        still_on_login_form = False

    if "index.php" in current_url and still_on_login_form:
        return False

    if "index.php" not in current_url:
        for frame in page.frames:
            try:
                if frame.locator("a:has-text('Log Out')").first.is_visible(timeout=1000):
                    return True
            except Exception:
                continue

    return False


def automate_anna_univ_login():
    register_number = "00000"
    dob = "00-00-0000"
    target_url = "https://coe.annauniv.edu/home/index.php"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=300)
        page = browser.new_page()

        page.on("dialog", lambda d: (print(f"[ALERT] {d.message}"), d.accept()))

        print("Navigating to Anna University Portal...")
        page.goto(target_url, timeout=60000, wait_until="domcontentloaded")
        time.sleep(2)

        max_attempts = 20
        attempt = 1
        logged_in = False

        while attempt <= max_attempts and not logged_in:
            print(f"\n--- Login Attempt {attempt}/{max_attempts} ---")
            print(f"URL before attempt: {page.url}")

            all_inputs = page.locator("input[type='text']").all()
            if len(all_inputs) < 5:
                print("Re-locating input fields...")
                page.goto(target_url, wait_until="domcontentloaded")
                time.sleep(2)
                all_inputs = page.locator("input[type='text']").all()

            student_reg = all_inputs[2]
            student_dob = all_inputs[3]
            student_captcha_box = all_inputs[4]

            # Fill Reg No & DOB
            student_reg.fill("")
            student_reg.press_sequentially(register_number, delay=80)

            student_dob.fill("")
            student_dob.press_sequentially(dob, delay=80)

            # Process Captcha Image
            student_captcha_img = page.locator("img[src*='captcha'], img[src*='php']").last
            captcha_raw_bytes = student_captcha_img.screenshot()

            cleaned_bytes = preprocess_captcha(captcha_raw_bytes)

            # Save both versions so you can visually inspect exactly what
            # OCR saw on this attempt — check captcha_debug/ after a run.
            with open(f"{DEBUG_DIR}/attempt_{attempt}_raw.png", "wb") as f:
                f.write(captcha_raw_bytes)
            with open(f"{DEBUG_DIR}/attempt_{attempt}_cleaned.png", "wb") as f:
                f.write(cleaned_bytes)

            ocr_results = reader.readtext(cleaned_bytes, detail=0, allowlist=OCR_ALLOWLIST)
            captcha_text = "".join(c for c in "".join(ocr_results) if c.isalnum()).strip()

            if len(captcha_text) != EXPECTED_CAPTCHA_LEN:
                # Try the raw (unprocessed) image as a fallback read
                raw_ocr = reader.readtext(captcha_raw_bytes, detail=0, allowlist=OCR_ALLOWLIST)
                raw_text = "".join(c for c in "".join(raw_ocr) if c.isalnum()).strip()
                if len(raw_text) == EXPECTED_CAPTCHA_LEN:
                    captcha_text = raw_text

            print(f"Extracted Captcha: '{captcha_text}' (len={len(captcha_text)})")

            if len(captcha_text) != EXPECTED_CAPTCHA_LEN:
                print(f"Captcha extraction invalid (expected {EXPECTED_CAPTCHA_LEN} chars). "
                      f"Reloading page for a fresh captcha... [saved to {DEBUG_DIR}/attempt_{attempt}_*.png]")

                page.reload(wait_until="domcontentloaded")
                time.sleep(2)
                attempt += 1
                continue

            # Type Captcha & Submit
            student_captcha_box.fill("")
            student_captcha_box.press_sequentially(captcha_text, delay=80)
            time.sleep(1)
            student_captcha_box.press("Enter")
            print("Form submitted. Checking page response...")

            # REALTIME VERIFICATION (fixed logic — see verify_login_success)
            result = verify_login_success(page)
            print(f"URL after verification: {page.url}")
            print(f"verify_login_success() -> {result}")

            if result:
                print("SUCCESS: Authenticated session confirmed! Student details page reached.")
                logged_in = True
                break
            else:
                print("FAILED: Login Rejected or stayed on login form. Retrying...")
                attempt += 1
                # No need to click here — the server already served a fresh
                # captcha as part of the page reload that just happened.
                time.sleep(2)

        if not logged_in:
            print("Could not log in after maximum attempts. Please check credentials or run again.")
            browser.close()
            return

        # Navigate to Exam Results Tab
        time.sleep(2)
        results_element = page.locator("a:has-text('EXAM RESULTS'), font:has-text('EXAM RESULTS')").first

        try:
            if results_element.is_visible(timeout=5000):
                results_element.click()
                print("Clicked EXAM RESULTS tab.")
            else:
                print("EXAM RESULTS tab not found on the authenticated page — not force-navigating "
                      "to avoid capturing a false screenshot.")
                browser.close()
                return
        except Exception as e:
            print(f"Could not locate EXAM RESULTS tab ({e}). Not force-navigating.")
            browser.close()
            return

        time.sleep(3)
        page.screenshot(path="scr2.png", full_page=True)
        print("Screenshot captured successfully and saved as 'scr.png'.")

        time.sleep(2)
        browser.close()


if __name__ == "__main__":
    automate_anna_univ_login()
