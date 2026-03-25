import os
import time
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Column, Boolean, Integer, Float
import sqlalchemy as sa

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

db = SQLAlchemy(app)

def run_migration():
    """Run the migration to add knowledge graph columns to the Document table."""
    print("Starting migration to add knowledge graph columns to Document table...")
    start_time = time.time()
    
    with app.app_context():
        # Get connection
        connection = db.engine.connect()
        
        # Check if the columns already exist
        inspector = sa.inspect(db.engine)
        columns = [column['name'] for column in inspector.get_columns('document')]
        
        # Add columns if they don't exist
        if 'kg_processed' not in columns:
            print("Adding 'kg_processed' column to Document table...")
            connection.execute(sa.text(
                "ALTER TABLE document ADD COLUMN kg_processed BOOLEAN DEFAULT FALSE"
            ))
        
        if 'kg_entity_count' not in columns:
            print("Adding 'kg_entity_count' column to Document table...")
            connection.execute(sa.text(
                "ALTER TABLE document ADD COLUMN kg_entity_count INTEGER"
            ))
        
        if 'kg_relationship_count' not in columns:
            print("Adding 'kg_relationship_count' column to Document table...")
            connection.execute(sa.text(
                "ALTER TABLE document ADD COLUMN kg_relationship_count INTEGER"
            ))
        
        if 'kg_processing_time' not in columns:
            print("Adding 'kg_processing_time' column to Document table...")
            connection.execute(sa.text(
                "ALTER TABLE document ADD COLUMN kg_processing_time FLOAT"
            ))
        
        # Commit the transaction
        connection.commit()
        connection.close()
    
    end_time = time.time()
    print(f"Migration completed in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    run_migration()