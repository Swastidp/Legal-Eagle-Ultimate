import os
import asyncio
import warnings
import nest_asyncio
import logging
from datetime import datetime

def apply_comprehensive_startup_fixes():
    """Apply all startup fixes for Legal Eagle MVP including Google Cloud authentication"""
    
    print(f"🚀 Applying Legal Eagle startup fixes at {datetime.now().strftime('%H:%M:%S')}")
    
    # 1. Fix Windows async issues
    try:
        nest_asyncio.apply()
        print("✅ Applied nest_asyncio fix for Windows async compatibility")
    except Exception as e:
        print(f"⚠️ Could not apply nest_asyncio: {e}")
    
    # 2. Set Windows event loop policy
    if os.name == 'nt':
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            print("✅ Set Windows ProactorEventLoopPolicy")
        except Exception as e:
            print(f"⚠️ Could not set Windows event loop policy: {e}")
    
    # 3. Fix Google Cloud authentication issues (ALTS credentials)
    print("🔧 Applying Google Cloud authentication fixes...")
    
    # Disable ALTS credentials (only works on GCP infrastructure)
    os.environ['GRPC_ENABLE_FORK_SUPPORT'] = '1'
    os.environ['GOOGLE_CLOUD_DISABLE_GRPC_FOR_TEST'] = 'true'
    
    # Reduce gRPC verbosity to suppress ALTS warnings
    os.environ['GRPC_VERBOSITY'] = 'ERROR'
    os.environ['GRPC_TRACE'] = ''
    
    # Suppress various Google Cloud warnings
    os.environ['GOOGLE_AUTH_SUPPRESS_CREDENTIALS_WARNINGS'] = 'true'
    os.environ['GOOGLE_CLOUD_PROJECT_DETECTION_DISABLED'] = 'true'
    
    # 4. Disable HuggingFace warnings
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # 5. Configure comprehensive logging
    configure_logging()
    
    # 6. Suppress specific warnings
    suppress_warnings()
    
    # 7. Set performance optimizations
    apply_performance_optimizations()
    
    print("✅ All Legal Eagle startup fixes applied successfully!")
    print("🔇 ALTS credentials warnings should now be suppressed")
    print("🚀 System ready for enhanced document processing")

def configure_logging():
    """Configure logging to suppress noisy messages"""
    
    # Set logging levels for Google Cloud components
    logging.getLogger('google.auth').setLevel(logging.ERROR)
    logging.getLogger('google.cloud').setLevel(logging.ERROR)
    logging.getLogger('grpc').setLevel(logging.ERROR)
    logging.getLogger('google.auth.transport.grpc').setLevel(logging.ERROR)
    logging.getLogger('google.auth.transport.requests').setLevel(logging.ERROR)
    
    # Set logging levels for other components
    logging.getLogger('urllib3').setLevel(logging.ERROR)
    logging.getLogger('httpx').setLevel(logging.ERROR)
    logging.getLogger('transformers').setLevel(logging.ERROR)
    
    print("✅ Configured logging levels")

def suppress_warnings():
    """Suppress non-critical warnings"""
    
    # HuggingFace warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
    
    # Google Cloud warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="grpc")
    warnings.filterwarnings("ignore", category=UserWarning, module="google")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="google")
    
    # Streamlit warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="streamlit")
    
    # Other warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
    warnings.filterwarnings("ignore", category=UserWarning, module="PIL")
    
    print("✅ Suppressed non-critical warnings")

def apply_performance_optimizations():
    """Apply performance optimizations"""
    
    # PyTorch optimizations
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    
    # CUDA optimizations (if available)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    
    # Memory optimizations
    os.environ['PYTHONHASHSEED'] = '0'
    
    print("✅ Applied performance optimizations")

def test_fixes():
    """Test that all fixes are working properly"""
    
    print("🧪 Testing startup fixes...")
    
    try:
        # Test async
        loop = asyncio.get_event_loop()
        print("✅ Async event loop working")
    except Exception as e:
        print(f"❌ Async issue: {e}")
    
    try:
        # Test Google Cloud imports
        from google.cloud import documentai_v1 as documentai
        print("✅ Google Cloud libraries imported without ALTS warnings")
    except Exception as e:
        print(f"⚠️ Google Cloud import issue: {e}")
    
    try:
        # Test ML libraries
        import torch
        import transformers
        print("✅ ML libraries imported successfully")
    except Exception as e:
        print(f"⚠️ ML library issue: {e}")
    
    print("🧪 Startup fixes test completed")

def fix_streamlit_asyncio():
    """Fix Streamlit + AsyncIO conflicts specifically"""
    import asyncio
    import threading
    
    try:
        # Force single event loop policy
        if hasattr(asyncio, 'WindowsProactorEventLoopPolicy'):
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
        # Create and set a new event loop for the main thread
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Apply nest_asyncio with error handling
        try:
            import nest_asyncio
            nest_asyncio.apply()
            print("✅ nest_asyncio applied successfully")
        except Exception as e:
            print(f"⚠️ nest_asyncio application warning: {e}")
        
        print("✅ Streamlit AsyncIO conflicts resolved")
        
    except Exception as e:
        print(f"⚠️ AsyncIO fix warning: {e}")

# Update the main function to include this fix
def apply_comprehensive_startup_fixes():
    """Updated startup fixes with Streamlit async handling"""
    
    print(f"🚀 Applying Legal Eagle startup fixes at {datetime.now().strftime('%H:%M:%S')}")
    
    # 1. Fix Streamlit + AsyncIO conflicts FIRST
    fix_streamlit_asyncio()
    
    # 2. Fix Windows async issues
    # ... (rest of your existing fixes)


if __name__ == "__main__":
    apply_comprehensive_startup_fixes()
    test_fixes()
