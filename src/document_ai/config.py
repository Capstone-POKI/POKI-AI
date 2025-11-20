import os
from dotenv import load_dotenv # type: ignore

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID", "pitchcoachai")
LOCATION = os.getenv("LOCATION", "us")
PROCESSORS = {
    "OCR": os.getenv("OCR_PROCESSOR_ID", "5a5219faee01df08"),
    "LAYOUT": os.getenv("LAYOUT_PROCESSOR_ID", "48046515c9645d3a"),
    "FORM": os.getenv("FORM_PROCESSOR_ID", "f788091cf561b641"),
}