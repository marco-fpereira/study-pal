import os
from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING

class MongoChatHistoryRepository:
    def __init__(self):
        load_dotenv()
        host = os.getenv("MONGO_HOST")
        port = os.getenv("MONGO_PORT")
        username = os.getenv("MONGO_USERNAME")
        password = os.getenv("MONGO_PASSWORD")

        self.__database_name = os.getenv("MONGO_DATABASE_NAME")
        self.__collection_name = os.getenv("COLLECTION_NAME")
        self.__connection_string = f"mongodb://{username}:{password}@{host}:{port}"

        try:
            self.__client = MongoClient(self.__connection_string)
            self.__collection = self.__client[self.__database_name][self.__collection_name]
            self._ensure_indexes()
        except Exception as e:
            print(f"Error trying to connect to MongoDB Database. Details: {e}")
            raise e


    def _ensure_indexes(self):
        # Fast session lookup
        self.__collection.create_index([("SessionId", ASCENDING)])

        # Auto-delete old sessions
        self.__collection.create_index(
            [("createdAt", ASCENDING)],
            expireAfterSeconds=604800  # 7 days
        )
    
    def get_client(self) -> MongoClient:
        return self.__client

    def get_database_name(self) -> str:
        return self.__database_name