import os
import logging
import psycopg2
import pandas as pd
from dotenv import load_dotenv

# ----------------------------
# Load Environment Variables
# ----------------------------
load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_PORT = os.getenv("DB_PORT")

# ----------------------------
# Configure Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Starting data extraction process...")

try:
    # ----------------------------
    # Connect to Database
    # ----------------------------
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASS
    )

    logging.info("Database connection Successful.")

    query = "SELECT * FROM ccod_bal;"
    df = pd.read_sql(query, conn)

    logging.info(f"Total rows fetched: {len(df)}")

    conn.close()
    logging.info("Database connection closed.")

except Exception as e:
    logging.error(f"Database error: {e}")
    raise


# ----------------------------
# Convert Row to Natural Text
# ----------------------------

def row_to_text(row):
    return (
        f"Account {row.accountno} belonging to {row.cust_name} "
        f"at branch {row.branch_name} has current balance {row.currentbalance} "
        f"with interest rate {row.intrate} percent. "
        f"Irregularity status is {row.irregularity}."
    )

logging.info("Converting rows to natural language...")

# Faster than iterrows()
texts = df.apply(row_to_text, axis=1).tolist()

logging.info(f"Converted {len(texts)} rows into text format.")

# ----------------------------
# Optional: Save for inspection
# ----------------------------
output_file = "converted_data.txt" 


with open(output_file, "w", encoding="utf-8") as f:
    for text in texts:
        f.write(text + "\n")

logging.info(f"Text data saved to {output_file}") 

logging.info("Process completed successfully.")
# print("TEXTS",texts)
print("OUTPUT FILE",output_file)
