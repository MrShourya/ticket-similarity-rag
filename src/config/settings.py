from dataclasses import dataclass


@dataclass
class ColumnMap:
    """
    Maps Excel column names to internal fields.
    Adjust here if Excel headers change.
    """

    ticket_id: str = "Number"
    state: str = "State"
    created: str = "Created"
    parent_case: str = "Parent Case"

    description: str = "Description"
    short_description: str = "Short Description"

    area: str = "Area"
    sub_area: str = "Sub Area"

    # Optional L2 enrichment fields
    request: str = "Request"
    response: str = "Response"
    api_name: str = "API Name"
    operation_name: str = "Operation Name"
    timestamp: str = "Timestamp"


@dataclass
class IngestConfig:
    """
    Config for ingestion pipeline
    """

    excel_path: str = "data/tickets.xlsx"
    sheet_name: str | None = None

    require_area: bool = False
    unknown_area_value: str = "UNKNOWN"

    max_request_chars: int = 4000
    max_response_chars: int = 4000