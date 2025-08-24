Serial Event Extraction
The extractor creates an event-based CSV report event_based_report.csv. Columns:

serial_number
event_type (flexible: any non-empty string; e.g., Acquisition, Disposition, Transfer, Return)
event_date
associated_name
associated_address
source_file
file_created
file_modified
Notes:

Each row in a source CSV can yield multiple event records (e.g., separate Acquisition and Disposition).
Source file creation and modification timestamps are captured from the filesystem and included per record.
Repository
Default branch: main
Remote: origin -> https://github.com/nicktishchenko/A_Int.git EOF
