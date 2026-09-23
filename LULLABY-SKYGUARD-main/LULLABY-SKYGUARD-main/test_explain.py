
import psycopg2
import os
from dotenv import load_dotenv
load_dotenv()
conn = psycopg2.connect(os.environ['DATABASE_URL'])
cur = conn.cursor()
cur.execute('EXPLAIN (ANALYZE, BUFFERS) SELECT MAX(time) FROM sensor_readings WHERE station_id = \'AWS-CHN-024\' AND source = \'live\';')
for row in cur.fetchall():
    print(row[0])

