import aiosqlite

DB_NAME = "history.db"

async def init_db():
    async with aiosqlite.connect(DB_NAME) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT,
                image_name TEXT,
                result TEXT,
                confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.commit()

async def save_history(type_: str, image_name: str, result: str, confidence: float):
    async with aiosqlite.connect(DB_NAME) as db:
        await db.execute(
            "INSERT INTO history (type, image_name, result, confidence) VALUES (?,?,?,?)",
            (type_, image_name, result, confidence)
        )
        await db.commit()

async def get_all_history():
    async with aiosqlite.connect(DB_NAME) as db:
        cursor = await db.execute("SELECT * FROM history ORDER BY created_at DESC")
        rows = await cursor.fetchall()
        return [dict(zip([col[0] for col in cursor.description], row)) for row in rows]