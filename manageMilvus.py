from sentence_transformers import SentenceTransformer
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import numpy as np
import torch
import argparse
import os
import re

# 상수 정의
MODEL_PATH = './local_multilingual_e5_large_instruct'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(24)

def get_collection_id(file_path):
    filename = os.path.basename(file_path)
    match = re.search(r'_(\d+)\.txt$', filename)
    if match:
        return match.group(1)
    raise ValueError(f"파일 이름 '{filename}'에서 ID를 추출할 수 없습니다. 형식은 'name_숫자.txt'여야 합니다.")

def connect_to_milvus():
    connections.connect("default", host="localhost", port="19530")

def load_texts_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def setup_collection(collection_id, dim=1024):
    collection_name = f"collection_{collection_id}"
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
    
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    
    schema = CollectionSchema(fields, description=f"Text and embedding storage collection for dataset {collection_id}")
    collection = Collection(name=collection_name, schema=schema)
    
    index_params = {
        "index_type": "GPU_IVF_PQ",
        "metric_type": "IP",
        "params": {
            "nlist": 2048,
            "m": 64,
            "nbits": 8,
            "gpu_id": [0, 1]
        }
    }
    
    collection.create_index(
        field_name="vector", 
        index_params=index_params
    )
    
    return collection

def insert_data(collection, texts):
    model = SentenceTransformer(MODEL_PATH)
    model.to(device)
    
    batch_size = 128
    total_texts = len(texts)
    
    for i in range(0, total_texts, batch_size):
        batch_texts = texts[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(total_texts + batch_size - 1)//batch_size}")
        
        embeddings = model.encode(
            batch_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=device
        )
        
        collection.insert([
            batch_texts,
            embeddings.tolist()
        ])
        
        print(f"Inserted {len(batch_texts)} texts")

def list_collections():
    """모든 컬렉션 목록을 출력합니다."""
    collections = utility.list_collections()
    if not collections:
        print("No collections found.")
    else:
        print("Available collections:")
        for collection in collections:
            print(f"- {collection}")

def delete_collection(collection_name):
    """특정 컬렉션을 삭제합니다."""
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"Collection '{collection_name}' has been deleted.")
    else:
        print(f"Collection '{collection_name}' does not exist.")

def search_collection(collection_name, query_text, top_k=5):
    """컬렉션에서 검색을 수행합니다."""
    if not utility.has_collection(collection_name):
        print(f"Collection '{collection_name}' does not exist.")
        return
    
    collection = Collection(collection_name)
    collection.load()
    
    model = SentenceTransformer(MODEL_PATH)
    model.to(device)
    
    query_vector = model.encode(
        query_text,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device=device
    ).tolist()
    
    search_params = {
        "metric_type": "IP",
        "params": {"nprobe": 16}
    }
    
    results = collection.search(
        data=[query_vector],
        anns_field="vector",
        param=search_params,
        limit=top_k,
        output_fields=["text"]
    )
    
    print(f"\nSearch results for: '{query_text}'")
    for i, hits in enumerate(results):
        print("\nTop matches:")
        for hit in hits:
            print(f"Text: {hit.entity.get('text')}")
            print(f"Distance: {hit.distance:.4f}\n")
    
    collection.release()

def main():
    parser = argparse.ArgumentParser(description='Milvus Collection Manager')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Insert command
    insert_parser = subparsers.add_parser('insert', help='Insert data into a new collection')
    insert_parser.add_argument('file_path', type=str, help='Path to the text file (format: name_number.txt)')
    
    # List command
    subparsers.add_parser('list', help='List all collections')
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete a collection')
    delete_parser.add_argument('collection_name', type=str, help='Name of the collection to delete')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search in a collection')
    search_parser.add_argument('collection_name', type=str, help='Name of the collection to search in')
    search_parser.add_argument('query', type=str, help='Search query text')
    search_parser.add_argument('--top-k', type=int, default=5, help='Number of results to return')
    
    args = parser.parse_args()
    
    try:
        connect_to_milvus()
        
        if args.command == 'insert':
            collection_id = get_collection_id(args.file_path)
            print(f"Using collection ID: {collection_id}")
            
            print(f"Loading texts from {args.file_path}...")
            texts = load_texts_from_file(args.file_path)
            print(f"Loaded {len(texts)} texts")
            
            print(f"Setting up collection_{collection_id}...")
            collection = setup_collection(collection_id)
            
            print("Starting data insertion...")
            insert_data(collection, texts)
            
            print("Loading collection...")
            collection.load()
            
            print("Data insertion completed successfully!")
            
        elif args.command == 'list':
            list_collections()
            
        elif args.command == 'delete':
            delete_collection(args.collection_name)
            
        elif args.command == 'search':
            search_collection(args.collection_name, args.query, args.top_k)
            
        else:
            parser.print_help()
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
