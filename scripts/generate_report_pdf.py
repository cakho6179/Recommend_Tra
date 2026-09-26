"""
Script sinh file PDF báo cáo chuẩn chỉnh từ file HTML bằng Playwright + Edge
Tự động chờ Mermaid JS render toàn bộ sơ đồ vector trước khi xuất PDF.
"""
import os
import sys
import asyncio
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from playwright.async_api import async_playwright

BASE_DIR = Path(__file__).resolve().parent.parent
HTML_FILE = BASE_DIR / "docs" / "reports" / "Personalized_Tea_Recommender_System_Report.html"
PDF_FILE = BASE_DIR / "docs" / "reports" / "Personalized_Tea_Recommender_System_Report.pdf"


async def generate_pdf():
    print("=" * 65)
    print("  SINH BÁO CÁO PDF: HƯƠNG VÂN TRÀ RECOMMENDER SYSTEM  ")
    print("=" * 65)
    
    if not HTML_FILE.exists():
        raise FileNotFoundError(f"Không tìm thấy file HTML: {HTML_FILE}")

    abs_html_uri = HTML_FILE.as_uri()
    print(f"[1/3] Đang nạp tài liệu HTML: {abs_html_uri} ...")

    async with async_playwright() as p:
        browser = await p.chromium.launch(channel="msedge", headless=True)
        page = await browser.new_page()

        # Nạp trang HTML
        await page.goto(abs_html_uri, wait_until="networkidle")

        print("[2/3] Đang chờ Mermaid JS render toàn bộ sơ đồ đồ thị vector...")
        # Đợi các thẻ SVG của mermaid xuất hiện
        try:
            await page.wait_for_selector(".mermaid svg", timeout=20000)
            print(" -> Đã render thành công các biểu đồ Mermaid.")
        except Exception as e:
            print(" -> Cảnh báo khi đợi Mermaid:", e)

        # Chờ thêm 2 giây để đảm bảo layout và font hoàn tất
        await asyncio.sleep(2.5)

        print(f"[3/3] Đang xuất file PDF sang: {PDF_FILE} ...")
        await page.pdf(
            path=str(PDF_FILE),
            format="A4",
            print_background=True,
            prefer_css_page_size=True,
        )
        await browser.close()

    if PDF_FILE.exists():
        file_size_kb = PDF_FILE.stat().st_size / 1024
        print(f"\n✅ ĐÃ XUẤT FILE PDF THÀNH CÔNG!")
        print(f" -> Đường dẫn: {PDF_FILE}")
        print(f" -> Kích thước: {file_size_kb:.1f} KB")
    else:
        raise RuntimeError("Xuất file PDF thất bại, không tìm thấy file đích.")


if __name__ == "__main__":
    asyncio.run(generate_pdf())
