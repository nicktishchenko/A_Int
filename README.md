This repository contains two Jupyter notebooks:
- image_processing.ipynb — image/OCR experiments and processing.
- serial_extractor.ipynb — event-based extraction of firearm serial numbers and associated data from CSVs.

## Setup

```bash
python3 -m venv allied
source allied/bin/activate
pip install -r requirements.txt
```

## Usage

Open the notebooks in Jupyter or VS Code:
```bash
jupyter notebook
# or
code image_processing.ipynb serial_extractor.ipynb
```

### Serial Event Extraction
The extractor produces an event-based CSV report `event_based_report.csv`. Columns:
- serial_number
- event_type (flexible: any non-empty string; e.g., Acquisition, Disposition, Transfer, Return)
- event_date
- associated_name
- associated_address
- source_file
- file_created
- file_modified

Notes:
- Each source row can yield multiple event records (e.g., separate Acquisition and Disposition).
- Source file creation and modification timestamps are captured from the filesystem and included per record.

## Repository
- Default branch: main
- Remote: origin -> https://github.com/nicktishchenko/A_Int.git

