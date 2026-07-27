from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from app.modules.ocr.regex_parser import CCCDRegexParser


def main() -> None:
    raw_text = [
        "CONG HOA XA HOI CHU NGHiA VIET NaM",
        "CAN CUOC CONC DAN",
        "Citizen Identity Card",
        "S6 | No:",
        "001095014159",
        "Hova ten / Full name:",
        "NGUYEN HOANG NAM",
        "Ngay sinh ! Date of birth;",
        "24/0311995",
        "Gioitinh / Sex:",
        "Nam",
        "Quoc tich / Netonelt",
        "Vict Nana",
    ]

    parser = CCCDRegexParser()
    result = parser.parse(raw_text)

    print(result)


if __name__ == "__main__":
    main()
