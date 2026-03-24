import clickhouse_connect
import os
from dotenv import load_dotenv

load_dotenv()

def test_clickhouse_connection():
    try:
        # Using the individual environment variables we added earlier
        host = os.getenv("CLICKHOUSE_HOST")
        port = int(os.getenv("CLICKHOUSE_PORT", 13535))
        user = os.getenv("CLICKHOUSE_USER")
        password = os.getenv("CLICKHOUSE_PASSWORD")
        
        print(f"Attempting to connect to ClickHouse at {host}:{port}...")
        
        client = clickhouse_connect.get_client(
            host=host,
            port=port,
            username=user,
            password=password,
            secure=True
        )
        
        result = client.command('SELECT 1')
        print(f"Success! Query result: {result}")
        
    except Exception as e:
        print(f"Failed to connect: {e}")

if __name__ == "__main__":
    test_clickhouse_connection()
