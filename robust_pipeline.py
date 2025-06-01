#!/usr/bin/env python3
"""
Enhanced pipeline with progress tracking and robust error handling
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.advanced_dataset_generator import AdvancedDatasetGenerator
from src.config import Config

def main():
    parser = argparse.ArgumentParser(description="Robust pipeline: documents → vector store → DPO dataset")
    parser.add_argument("--input_dir", 
                      default=Config.DEFAULT_INPUT_DIR,
                      help="Directory containing documents to process")
    parser.add_argument("--output_dir", 
                      default="data",
                      help="Base output directory")
    parser.add_argument("--num_samples", 
                      type=int, 
                      default=500,
                      help="Number of DPO samples to generate")
    parser.add_argument("--chunk_size", 
                      type=int, 
                      default=Config.CHUNK_SIZE,
                      help="Size of text chunks")
    parser.add_argument("--skip_vectorstore", 
                      action="store_true",
                      help="Skip vector store creation (use existing)")
    parser.add_argument("--max_retries", 
                      type=int, 
                      default=Config.MAX_RETRIES,
                      help="Maximum API retries")
    parser.add_argument("--batch_size", 
                      type=int, 
                      default=10,
                      help="Batch size for embeddings API")
    parser.add_argument("--quality_target", 
                      choices=["high", "ultra", "production"],
                      default="high",
                      help="Quality target for dataset generation")
    parser.add_argument("--enable_difficulty_balancing", 
                      action="store_true",
                      help="Enable difficulty distribution balancing")
    parser.add_argument("--min_query_score", 
                      type=float, 
                      default=0.7,
                      help="Minimum query quality score")
    parser.add_argument("--min_response_score", 
                      type=float, 
                      default=0.75,
                      help="Minimum response quality score")
    
    args = parser.parse_args()
    
    start_time = datetime.now()
    
    try:
        # Validate configuration
        Config.validate()
        
        print("🚀 Starting Enhanced DPO Dataset Generation Pipeline")
        print("=" * 60)
        print(f"⏰ Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📂 Input directory: {args.input_dir}")
        print(f"📁 Output directory: {args.output_dir}")
        print(f"🎯 Target samples: {args.num_samples}")
        print(f"📏 Chunk size: {args.chunk_size}")
        print(f"🔄 Max retries: {args.max_retries}")
        print(f"📦 Batch size: {args.batch_size}")
        print()
        
        # Set up paths
        vectorstore_dir = Path(args.output_dir) / "vectorstore"
        dataset_file = Path(args.output_dir) / "dpo_dataset.json"
        
        # Create output directory
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        vector_store = VectorStore()
        
        if not args.skip_vectorstore:
            # Step 1: Process documents
            print("📚 Step 1: Processing documents...")
            processor = DocumentProcessor()
            processor.chunk_size = args.chunk_size
            
            documents = processor.process_directory(args.input_dir)
            
            if not documents:
                print("❌ No documents found to process!")
                return
            
            print(f"✅ Processed {len(documents)} document chunks")
            
            # Show document statistics
            total_chars = sum(doc['char_count'] for doc in documents)
            avg_chars = total_chars / len(documents)
            print(f"📊 Document stats: {total_chars:,} total chars, {avg_chars:.0f} avg per chunk")
            print()
            
            # Step 2: Create vector store with robust API handling
            print("🔍 Step 2: Creating vector store...")
            print(f"🌐 Using robust API with {args.max_retries} max retries")
            
            # Use custom embeddings API settings
            from src.embeddings_api import EmbeddingsAPI
            robust_api = EmbeddingsAPI(max_retries=args.max_retries)
            vector_store.embeddings_api = robust_api
            
            vector_store.create_from_documents(documents)
            vector_store.save(str(vectorstore_dir))
            
            # Print vector store stats
            stats = vector_store.get_stats()
            print("✅ Vector store statistics:")
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    print(f"    {key}: {value:,}")
                else:
                    print(f"    {key}: {value}")
            print()
        else:
            # Load existing vector store
            print("📂 Loading existing vector store...")
            vector_store.load(str(vectorstore_dir))
            stats = vector_store.get_stats()
            print(f"✅ Loaded vector store with {stats.get('num_documents', 0):,} documents")
            print()        # Step 3: Generate DPO dataset using Advanced Generator
        print("🎯 Step 3: Generating advanced DPO dataset...")
        generator = AdvancedDatasetGenerator(vector_store)
        
        # Configure advanced settings
        if hasattr(args, 'min_query_score'):
            generator.min_query_score = args.min_query_score
        if hasattr(args, 'min_response_score'):
            generator.min_response_score = args.min_response_score
        if hasattr(args, 'enable_difficulty_balancing'):
            generator.enable_difficulty_balancing = args.enable_difficulty_balancing
        
        quality_target = getattr(args, 'quality_target', 'high')
        dataset = generator.generate_dataset(num_samples=args.num_samples, quality_target=quality_target)
        
        if not dataset:
            print("❌ No dataset samples generated!")
            return
        
        # Save dataset
        generator.save_dataset(dataset, str(dataset_file))
        
        # Calculate execution time
        end_time = datetime.now()
        execution_time = end_time - start_time
          # Print final statistics
        dataset_stats = generator.get_advanced_stats(dataset)
        print("\n🎉 Pipeline completed successfully!")
        print("=" * 60)
        print(f"⏱️  Total execution time: {execution_time}")
        print("📊 Final Statistics:")
        for key, value in dataset_stats.items():
            if isinstance(value, float):
                print(f"    {key}: {value:.2f}")
            elif isinstance(value, int):
                print(f"    {key}: {value:,}")
            elif isinstance(value, dict):
                print(f"    {key}: {value}")
            else:
                print(f"    {key}: {value}")
        
        print(f"\n📁 Files created:")
        print(f"    Vector store: {vectorstore_dir}")
        print(f"    DPO dataset: {dataset_file}")
        
        # Show sample entries
        print(f"\n📋 Sample DPO entries:")
        for i, sample in enumerate(dataset[:2]):
            print(f"\n--- Muestra {i+1} ---")
            print(f"Consulta: {sample['query']}")
            print(f"Respuesta buena: {sample['good_response'][:100]}...")
            print(f"Respuesta mala: {sample['bad_response'][:100]}...")
        
        print(f"\n✨ ¡Listo para entrenamiento DPO!")
        print(f"🇪🇸 Dataset generado en español desde documentos locales")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
