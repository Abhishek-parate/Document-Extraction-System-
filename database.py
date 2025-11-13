from models import db, User, Document
from flask import Flask

def init_db(app):
    with app.app_context():
        db.create_all()
        print("Database initialized successfully!")