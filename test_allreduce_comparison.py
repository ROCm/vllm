#!/usr/bin/env python3
"""
Test script to verify the allreduce comparison functionality.
This script demonstrates how the comparison works in the ROCm allreduce dispatcher.
"""

import torch
import torch.distributed as dist
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_allreduce_comparison():
    """Test the allreduce comparison functionality."""
    
    # Create test data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = torch.randn(1024, 1024, dtype=torch.float16, device=device)
    
    logger.info(f"Test input shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")
    logger.info(f"Input range: [{torch.min(input_tensor).item():.6f}, {torch.max(input_tensor).item():.6f}]")
    
    # Simulate the comparison logic
    # This would normally be done by the ROCm allreduce dispatcher
    custom_out = input_tensor.clone()
    torch_out = input_tensor.clone()
    
    # Simulate custom allreduce (in real scenario, this would be the actual custom op)
    # For testing, we'll just add some small random noise to simulate differences
    noise = torch.randn_like(input_tensor) * 1e-6
    custom_out = custom_out + noise
    
    # Calculate differences
    diff = custom_out - torch_out
    max_diff = torch.max(torch.abs(diff)).item()
    mean_diff = torch.mean(torch.abs(diff)).item()
    mse = torch.mean(diff ** 2).item()
    
    # Log results
    logger.info(f"AllReduce Comparison - Op: CustomAllReduce")
    logger.info(f"Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}, MSE: {mse:.6e}")
    logger.info(f"Input shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")
    
    # Check if differences are significant
    if max_diff > 1e-5:
        logger.warning(f"Significant difference detected! Max diff: {max_diff:.6e}")
        logger.warning(f"Custom op result shape: {custom_out.shape}, dtype: {custom_out.dtype}")
        logger.warning(f"Torch dist result shape: {torch_out.shape}, dtype: {torch_out.dtype}")
        
        # Additional debugging info
        logger.warning(f"Input range: [{torch.min(input_tensor).item():.6f}, {torch.max(input_tensor).item():.6f}]")
        logger.warning(f"Custom op range: [{torch.min(custom_out).item():.6f}, {torch.max(custom_out).item():.6f}]")
        logger.warning(f"Torch dist range: [{torch.min(torch_out).item():.6f}, {torch.max(torch_out).item():.6f}]")
    else:
        logger.info(f"AllReduce results match within tolerance (max diff: {max_diff:.6e})")

if __name__ == "__main__":
    test_allreduce_comparison()
