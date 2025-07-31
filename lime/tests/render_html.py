import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from PIL import Image
# from test_lime_parser import *
import time
import shap
# from lime_text_parser import SavedExplanation
import re
import matplotlib.pyplot as plt

# ==== CONFIG ====
HTML_DIR = r"./HTML_results/"
OUTPUT_DIR = os.path.join(HTML_DIR, "html_images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DIV_SELECTORS = [
    # "div.lime.predict_proba",
    # "div.lime.explanation",
    "div.lime.text_div"
]

# ==== SETUP HEADLESS CHROME ====
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--window-size=1600,1200")
chrome_options.add_argument("--force-device-scale-factor=2")
driver = webdriver.Chrome(options=chrome_options)

# def print_shap_images(exp_path):
#     pattern = re.compile(".*Shap.*")
#     for filename in os.listdir(exp_path):
#         if pattern.match(filename):
#             exp = SavedExplanation(filename).get_exp()
#             shap.plots.text(exp)
#             shap.plots.bar(exp)

def crop_whitespace(input_path, output_path, bg_color=(255, 255, 255, 255)):
    image = Image.open(input_path).convert("RGBA")

    # Get image data as a sequence of pixels
    bbox = image.getbbox()

    if image.mode == "RGBA":
        # Remove transparent or white borders
        datas = image.getdata()

        # Find non-white, non-transparent pixels
        non_empty_pixels = [
            (x % image.width, x // image.width)
            for x, pixel in enumerate(datas)
            if pixel[:3] != bg_color[:3] and pixel[3] != 0
        ]

        if non_empty_pixels:
            min_x = min(x for x, y in non_empty_pixels)
            max_x = max(x for x, y in non_empty_pixels)
            min_y = min(y for x, y in non_empty_pixels)
            max_y = max(y for x, y in non_empty_pixels)
            image = image.crop((min_x, min_y, max_x + 1, max_y + 1))

    image.save(output_path)


# ==== RENDER FUNCTION ====
def render_div_by_selector(selector, output_path, html_file):
    url = "file://" + os.path.abspath(html_file)
    driver.get(url)

    try:
        WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, selector))
        )

        element = driver.find_element(By.CSS_SELECTOR, selector)
        location = element.location_once_scrolled_into_view
        time.sleep(0.2)

        # Expand element height to fit full scrollable content
        driver.execute_script("""
            const el = arguments[0];
            el.style.height = el.scrollHeight + "px";
            el.style.overflow = "visible";
            el.style.maxHeight = 'none';
        """, element)

        # Re-fetch element size and location after expanding
        time.sleep(0.2)
        location = element.location_once_scrolled_into_view
        size = element.size

        driver.save_screenshot("temp_full_page.png")

        im = Image.open("temp_full_page.png")
        DPR = 0.5  # Device Pixel Ratio
        left = int(location["x"] * DPR)
        top = int(location["y"] * DPR)
        right = left + int(size["width"] * DPR)
        bottom = top + int(size["height"] * DPR)
        im = im.crop((left, top, right, bottom))
        im.save(output_path)
        print(f"✅ Saved {output_path}")

    except Exception as e:
        print(f"❌ Failed to find {selector} in {html_file}: {e}")

DISTINGUISHER = "Human"

# ==== MAIN LOOP ====
pattern = re.compile(f".*{re.escape(DISTINGUISHER)}.*")

for filename in os.listdir(HTML_DIR):
    if not (filename.endswith(".html") and pattern.match(filename)):
        continue

    if filename.find("shap") == -1:
        file_path = os.path.join(HTML_DIR, filename)
        base_name = os.path.splitext(filename)[0]

        for selector in DIV_SELECTORS:
            # Sanitize filename from selector
            safe_selector = selector.replace(".", "_").replace(" ", "")
            output_path = os.path.join(OUTPUT_DIR, f"{base_name}_{safe_selector}.png")

            render_div_by_selector(selector, output_path, file_path)
            crop_whitespace(output_path, output_path)
            crop_whitespace(output_path, output_path, bg_color=(0,0,0,0))

driver.quit()
