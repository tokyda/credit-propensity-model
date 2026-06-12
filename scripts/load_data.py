# scripts/load_data.py
import duckdb
from pathlib import Path

DB_PATH = 'data/berka.duckdb'
RAW_PATH = 'data/raw'

TABLES = {
    'account':  'account.csv',
    'client':   'client.csv',
    'disp':     'disp.csv',
    'trans':    'trans.csv',
    'loan':     'loan.csv',
    'card':     'card.csv',
    'district': 'district.csv',
    'order_':   'order.csv',
}


def load_raw_tables():
    conn = duckdb.connect(DB_PATH)
    for table_name, filename in TABLES.items():
        filepath = f"{RAW_PATH}/{filename}"
        conn.execute(f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT * FROM read_csv_auto('{filepath}', delim=';', header=true)
        """)
        count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        print(f"  {table_name}: {count:,} rows")
    conn.close()
    print("Done. Raw tables loaded into", DB_PATH)


def export_funnel(db_path=DB_PATH):
    """Run after dbt — exports mart_funnel to outputs/mart_funnel.csv."""
    conn = duckdb.connect(db_path, read_only=True)
    df = conn.execute("SELECT * FROM mart_funnel").df()
    conn.close()
    Path('outputs').mkdir(exist_ok=True)
    df.to_csv('outputs/mart_funnel.csv', index=False)
    print("Exported outputs/mart_funnel.csv")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'export-funnel':
        export_funnel()
    else:
        load_raw_tables()
