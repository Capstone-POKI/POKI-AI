import os
from dotenv import load_dotenv # type: ignore

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID", "pitchcoachai")
LOCATION = os.getenv("LOCATION", "us")
PROCESSORS = {
    "OCR": os.getenv("OCR_PROCESSOR_ID", "e41bb5d1cae96184"),
    "LAYOUT": os.getenv("LAYOUT_PROCESSOR_ID", "82698693210d7aa8"),
    "FORM": os.getenv("FORM_PROCESSOR_ID", "662d7f1f1e179648"),
}