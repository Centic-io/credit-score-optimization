import os

from dotenv import load_dotenv
from textwrap import dedent

load_dotenv()
class MongoDBConfig:
    USERNAME = os.environ.get("MONGO_USERNAME") or "just_for_dev"
    PASSWORD = os.environ.get("MONGO_PASSWORD") or "password_for_dev"
    HOST = os.environ.get("MONGO_HOSTS") or "localhost"
    PORT = os.environ.get("MONGO_PORT") or "27017"
    DATABASE = os.environ.get("MONGO_DATABASE") or "example_db"