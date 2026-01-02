import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from .text_chunking import ComplaintTextSplitter
from .embeddings import ComplaintEmbedder
from .vector_store import ComplaintVectorStore

# Setup logging
def setup_logging():
    log_dir = Path(__file__).parent.parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f'complaint_processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def process_complaints(df: pd.DataFrame):
    logger = setup_logging()
    logger.info('Starting complaint processing pipeline')
    
    # Initialize components
    logger.info('Initializing components...')
    splitter = ComplaintTextSplitter(chunk_size=500, chunk_overlap=50)
    embedder = ComplaintEmbedder()
    vector_store = ComplaintVectorStore()
    
    # Process in batches to handle memory efficiently
    batch_size = 1000
    total_rows = len(df)
    logger.info(f'Processing {total_rows} rows in batches of {batch_size}')
    
    for batch_idx, start_idx in enumerate(range(0, total_rows, batch_size)):
        end_idx = min(start_idx + batch_size, total_rows)
        batch_df = df.iloc[start_idx:end_idx]
        
        # 1. Chunking
        logger.info(f'Batch {batch_idx + 1}: Splitting text into chunks...')
        chunks = splitter.split_dataframe(batch_df, 'consumer_complaint_narrative')
        
        # 2. Embedding
        logger.info(f'Batch {batch_idx + 1}: Generating embeddings...')
        texts = [chunk['text'] for chunk in chunks]
        embeddings = embedder.embed_texts(texts)
        
        # 3. Vector Store
        logger.info(f'Batch {batch_idx + 1}: Storing embeddings...')
        metadatas = [chunk['metadata'] for chunk in chunks]
        vector_store.add_embeddings(embeddings, metadatas)
        
        logger.info(f'Completed batch {batch_idx + 1}: Processed rows {start_idx}-{end_idx}')
    
    logger.info('Processing complete. Vector store created at vector_store/')
    
if __name__ == "__main__":
    try:
        # Load cleaned data
        data_path = Path(__file__).parent.parent.parent / 'data' / 'filtered_complaints.csv'
        logger.info(f'Loading data from {data_path}')
        df = pd.read_csv(data_path)
        
        # Process the data
        process_complaints(df)
        
    except Exception as e:
        logger.error(f'Error during processing: {str(e)}', exc_info=True)