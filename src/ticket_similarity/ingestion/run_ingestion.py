from ticket_similarity.config.settings import ColumnMap, IngestConfig
from ticket_similarity.ingestion.ingest import ingest_excel, export_results


def main():
    cfg = IngestConfig(excel_path="data/tickets.xlsx", sheet_name=None)
    cols = ColumnMap()

    tickets = ingest_excel(cfg, cols)

    print(f"Processed {len(tickets)} tickets")

    export_results(tickets)


if __name__ == "__main__":
    main()