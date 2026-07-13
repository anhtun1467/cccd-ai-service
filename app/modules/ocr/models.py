from dataclasses import dataclass


@dataclass
class OCRTextBox:
    text: str
    confidence: float
    box: list


@dataclass
class OCRResult:
    success: bool
    message: str
    raw_text: list[str]
    text_boxes: list[OCRTextBox]