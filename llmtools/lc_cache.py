# Caching
from langchain.cache import InMemoryCache
from langchain.cache import SQLiteCache
from langchain.globals import set_llm_cache

def setup_inmemory_cache():
    set_llm_cache(InMemoryCache())

def setup_sqlite_cache(db_cache_name=".langchain.db"):
    set_llm_cache(SQLiteCache(database_path=db_cache_name))
