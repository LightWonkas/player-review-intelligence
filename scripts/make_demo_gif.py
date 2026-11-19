# /// script
# requires-python = ">=3.12"
# dependencies = ["pillow>=11"]
# ///
"""Make the README demo GIF from real dashboard screenshots (run by hand, never in CI).

    uv run review-intel dashboard            # in another terminal
    uv run scripts/make_demo_gif.py          # Pillow is installed just for this script

Takes one tall headless Chrome/Edge screenshot of the running dashboard and scrolls
through it frame by frame. (Screenshotting #anchors directly is flaky in headless
mode: the page is sometimes captured half-painted.) Writes docs/demo.gif and keeps
the first frame as docs/dashboard.png.
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome",
    "chromium",
    "microsoft-edge",
]
WIDTH, VIEW, TALL = 1280, 860, 5000
OUT_WIDTH = 960
# (fraction of the scrollable height, seconds on screen): tiles and issues, charts,
# report summary, report issues
STOPS = [(0.0, 3.0), (0.21, 4.0), (0.38, 3.0), (0.56, 3.0)]


def find_browser(explicit: str | None) -> str:
    for candidate in [explicit, *CANDIDATES]:
        if candidate and (Path(candidate).exists() or shutil.which(candidate)):
            return candidate
    raise SystemExit("No Chrome or Edge found; pass --browser /path/to/chrome")


def screenshot(browser: str, url: str, out: Path) -> None:
    subprocess.run(
        [
            browser,
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            f"--window-size={WIDTH},{TALL}",
            "--virtual-time-budget=4000",
            f"--screenshot={out}",
            url,
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )


def content_height(image: Image.Image) -> int:
    """Last row that differs from the page background (the tall shot has empty space below)."""
    background = image.getpixel((5, image.height - 5))
    for y in range(image.height - 1, 0, -8):
        row = image.crop((0, y, image.width, y + 1)).getcolors(maxcolors=4096) or []
        if any(color != background for _, color in row):
            return int(min(image.height, y + 48))
    return int(image.height)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8765/apps/critical-strike")
    parser.add_argument("--browser")
    args = parser.parse_args()
    browser = find_browser(args.browser)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "page.png"
        screenshot(browser, args.url, path)
        page = Image.open(path).convert("RGB")
        page.load()
    height = content_height(page)
    scroll = max(0, height - VIEW)
    frames, durations = [], []
    for fraction, seconds in STOPS:
        top = int(scroll * fraction)
        frame = page.crop((0, top, WIDTH, top + VIEW))
        frames.append(frame.resize((OUT_WIDTH, OUT_WIDTH * VIEW // WIDTH), Image.LANCZOS))
        durations.append(int(seconds * 1000))

    DOCS.mkdir(exist_ok=True)
    frames[0].save(DOCS / "dashboard.png", optimize=True)
    quantized = [f.quantize(colors=256, method=Image.Quantize.MEDIANCUT) for f in frames]
    quantized[0].save(
        DOCS / "demo.gif",
        save_all=True,
        append_images=quantized[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    for name in ("demo.gif", "dashboard.png"):
        size = (DOCS / name).stat().st_size // 1024
        print(f"wrote docs/{name} ({size} KB, page {height}px tall)")


if __name__ == "__main__":
    main()
