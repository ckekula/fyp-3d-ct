"""Backend entrypoint for the DICOM test script.

Run this from the backend folder with:
python test.py
"""

from app.services.test import main


if __name__ == "__main__":
    main()