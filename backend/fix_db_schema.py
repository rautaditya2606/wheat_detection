import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def fix_schema():
    print(f"Connecting to: {DATABASE_URL}")
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        print("Checking columns in 'feedback'...")
        try:
            # Check if column exists
            result = conn.execute(text("DESCRIBE TABLE default.feedback"))
            columns = [row[0] for row in result]
            print(f"Found columns: {columns}")
            
            if 'used_in_training' not in columns:
                print("Adding 'used_in_training' column...")
                conn.execute(text("ALTER TABLE default.feedback ADD COLUMN used_in_training UInt8 DEFAULT 0"))
                print("Success!")
            
            if 'is_verified' not in columns:
                print("Adding 'is_verified' column...")
                conn.execute(text("ALTER TABLE default.feedback ADD COLUMN is_verified UInt8 DEFAULT 0"))
                print("Success!")
            
            if 'confidence' not in columns:
                print("Adding 'confidence' column...")
                conn.execute(text("ALTER TABLE default.feedback ADD COLUMN confidence Float32 DEFAULT 0.0"))
                print("Success!")
            else:
                print("Columns up to date.")
                
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    fix_schema()
