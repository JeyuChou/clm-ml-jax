"""GPU configuration utilities for JAX.

This module provides utilities to configure and check GPU availability for JAX.
"""

import os
import jax
from typing import Optional, List


def check_gpu_available() -> bool:
    """Check if GPU is available for JAX.
    
    Returns:
        True if GPU is available, False otherwise
    """
    try:
        devices = jax.devices()
        return devices[0].platform == 'gpu'
    except Exception:
        return False


def get_gpu_devices() -> List:
    """Get list of available GPU devices.
    
    Returns:
        List of GPU devices, empty if no GPUs available
    """
    try:
        return jax.devices('gpu')
    except Exception:
        return []


def configure_gpu(
    preallocate: bool = False,
    allocator: str = 'platform',
    visible_devices: Optional[str] = None,
) -> bool:
    """Configure JAX to use GPU if available.
    
    Args:
        preallocate: If True, preallocate GPU memory (default: False)
        allocator: Memory allocator ('platform' or 'default')
        visible_devices: Comma-separated list of GPU indices to use (e.g., "0,1")
        
    Returns:
        True if GPU is configured and available, False otherwise
    """
    # Set visible devices if specified
    if visible_devices is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices
    
    # Configure memory allocation
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true' if preallocate else 'false'
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = allocator
    
    # Check if GPU is available
    if check_gpu_available():
        print(f"✓ GPU is available: {jax.devices()[0]}")
        print(f"  JAX version: {jax.__version__}")
        print(f"  Default backend: {jax.default_backend()}")
        return True
    else:
        print("✗ GPU not available, using CPU")
        print(f"  Available devices: {jax.devices()}")
        return False


def print_device_info():
    """Print information about available JAX devices."""
    print("=" * 60)
    print("JAX Device Information")
    print("=" * 60)
    print(f"JAX version: {jax.__version__}")
    print(f"Default backend: {jax.default_backend()}")
    print(f"Number of devices: {len(jax.devices())}")
    print()
    
    for i, device in enumerate(jax.devices()):
        print(f"Device {i}:")
        print(f"  Platform: {device.platform}")
        print(f"  Device ID: {device.id}")
        print(f"  Device: {device}")
        print()
    
    # Check for GPUs specifically
    gpu_devices = get_gpu_devices()
    if gpu_devices:
        print(f"GPU devices found: {len(gpu_devices)}")
        for gpu in gpu_devices:
            print(f"  {gpu}")
    else:
        print("No GPU devices found")
    print("=" * 60)


# Auto-configure on import (optional - comment out if you want manual control)
# USE_GPU = configure_gpu()
# DEFAULT_DEVICE = jax.devices()[0] if check_gpu_available() else jax.devices()[0]

if __name__ == "__main__":
    # Print device info when run as script
    print_device_info()
    
    # Try to configure GPU
    print("\nAttempting to configure GPU...")
    configure_gpu()

