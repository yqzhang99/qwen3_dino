# coding=utf-8

import argparse
import os
import shutil
import sys
from pathlib import Path
import torch

# Add project root to Python path
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from transformers.models.qwen3_vl import Qwen3VLConfig, Qwen3VLProcessor
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.dinov3_vit.configuration_dinov3_vit import DINOv3ViTConfig
from configuration_qwen3vl_dinov3vit import Qwen3VLDINOv3ViTConfig
from modeling_qwen3vl_dinov3vit import Qwen3VLForConditionalGenerationWithDINOv3
from weight_loading_utils import WeightLoadingManager


def copy_file_if_exists(src_path: str, dst_path: str, file_name: str):
    """Copy a file from source to destination if it exists."""
    src_file = os.path.join(src_path, file_name)
    dst_file = os.path.join(dst_path, file_name)
    
    if os.path.exists(src_file):
        shutil.copy2(src_file, dst_file)
        print(f"Copied {file_name} from {src_path} to {dst_path}")
        return True
    else:
        print(f"Warning: {file_name} not found in {src_path}, skipping...")
        return False


def save_qwen3vl_dinov3vit_checkpoint(
    qwen3vl_model_path: str,
    dinov3_model_path: str,
    output_dir: str,
    checkpoint_path: str = None,
    num_cameras: int = 6,
    num_timesteps: int = 2,
    num_scene_tokens: int = 900,
    flex_hidden_dim: int = 768,
):
    """
    Build Qwen3VL-DINOv3 mixed model, load partial weights, and save as Hugging Face format.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Build model and load Qwen3VL partial weights
    print("=" * 80)
    print("Step 1: Building Qwen3VL-DINOv3 mixed model and loading partial weights...")
    print("=" * 80)

    # Create or load configuration
    try:
        qwen3vl_config = Qwen3VLConfig.from_pretrained(qwen3vl_model_path)
    except Exception as e:
        print(f"Error loading Qwen3VL config: {e}")
        return

    # ==============================================================================
    # [FIX 1] Manually inject mRoPE config into TEXT_CONFIG if missing
    # ==============================================================================
    # 关键修改：必须检查并注入到 qwen3vl_config.text_config，而不仅仅是 root config
    
    # 1. 确定我们要操作的目标 Config 对象 (优先使用 text_config)
    target_config = getattr(qwen3vl_config, "text_config", qwen3vl_config)
    
    if not hasattr(target_config, "rope_scaling") or target_config.rope_scaling is None:
        print(f"Warning: Input model text_config missing rope_scaling. Injecting default Qwen3-VL mRoPE config.")
        
        # 使用你提供的标准配置 [24, 20, 20]
        mrope_config = {
            "mrope_interleaved": True,
            "mrope_section": [24, 20, 20], 
            "rope_type": "default",
            "type": "default"
        }
        
        # 注入到 text_config
        target_config.rope_scaling = mrope_config
        
        # 为了保险，也注入到 root config (某些校验逻辑可能会看)
        qwen3vl_config.rope_scaling = mrope_config
    else:
        print("Info: rope_scaling found in config.")

    # ==============================================================================
    # [FIX 2] Manually inject Vision Config if using defaults (Prevents OOM)
    # ==============================================================================
    if not hasattr(qwen3vl_config, "vision_config") or qwen3vl_config.vision_config is None:
        print("Warning: Input model is a text model (vision_config is None). Injecting default Qwen2-VL-2B vision config.")
        qwen3vl_config.vision_config = Qwen3VLVisionConfig(
            depth=32,
            embed_dim=1152,      
            hidden_size=1152,    
            hidden_act="quick_gelu",
            mlp_ratio=4,
            num_heads=16,
            in_channels=3,
            patch_size=14,
            spatial_merge_size=2,
            spatial_patch_size=14,
            temporal_patch_size=2,
        )
    else:
        v_conf = qwen3vl_config.vision_config
        v_hidden = getattr(v_conf, "hidden_size", 1152) if v_conf else 1152
        print(f"Using existing vision config with hidden_size: {v_hidden}")

    dinov3_config = DINOv3ViTConfig.from_pretrained(dinov3_model_path)

    flex_config_dict = {
        "num_cameras": num_cameras,
        "num_timesteps": num_timesteps,
        "num_scene_tokens": num_scene_tokens,
        "hidden_dim": flex_hidden_dim,
        "backbone_dim": dinov3_config.hidden_size,
        "llm_dim": qwen3vl_config.hidden_size,
        "num_layers": 8, 
        "dropout": 0.1,
    }
    print(f"Injecting Flex Config: {flex_config_dict}")
    
    config = Qwen3VLDINOv3ViTConfig.from_configs(
        qwen3vl_config=qwen3vl_config,
        dinov3_config=dinov3_config,
        flex_config=flex_config_dict
    )
    
    # [FIX 3] Use bfloat16 to save memory during initialization
    print("Initializing model with torch.bfloat16 to save memory...")
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    
    try:
        model = Qwen3VLForConditionalGenerationWithDINOv3(config)
    finally:
        torch.set_default_dtype(original_dtype)
    
    # Load Qwen3VL partial weights
    print("Loading weights...")
    weight_manager = WeightLoadingManager(model=model, qwen3vl_path=qwen3vl_model_path, dinov3_path=dinov3_model_path)
    missing_keys, unexpected_keys = weight_manager.load_mixed_components(strict=False)

    print(f"Loaded Qwen3VL partial weights. Missing keys: {len(missing_keys)} (expected: blocks only)")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")
    
    # Optionally load fine-tuned checkpoint
    if checkpoint_path is not None:
        print(f"\nLoading fine-tuned checkpoint from {checkpoint_path}...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
            else:
                state_dict = checkpoint
            
            new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            
            load_result = model.load_state_dict(new_state_dict, strict=False)
            print(f"Checkpoint loaded. Missing keys: {len(load_result.missing_keys)}")
            
            flex_keys_loaded = [k for k in new_state_dict.keys() if "flex_encoder" in k]
            if flex_keys_loaded:
                print(f"SUCCESS: Found {len(flex_keys_loaded)} Flex parameters in checkpoint.")
            else:
                print("\n!!! WARNING !!!: No 'flex_encoder' keys found. FlexEncoder is RANDOM.")

        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    else:
        print("\n!!! WARNING !!!: No checkpoint_path provided. FlexEncoder is RANDOM.")
    
    # 2. Save model
    print("\n" + "=" * 80)
    print(f"Step 2: Saving model to {output_dir}...")
    print("=" * 80)
    
    model.save_pretrained(output_dir, safe_serialization=True)
    
    try:
        processor = Qwen3VLProcessor.from_pretrained(qwen3vl_model_path)
        processor.save_pretrained(output_dir)
        print("Processor saved successfully!")
    except Exception as e:
        print(f"Warning: Could not load/save Qwen3VLProcessor: {e}")
        print("You may need to manually copy preprocessor_config.json from a valid Qwen2-VL repo.")

    # 3. Copy files
    print("\n" + "=" * 80)
    print("Step 3: Copying auxiliary files...")
    
    src_path = qwen3vl_model_path
    for filename in ["generation_config.json", "tokenizer_config.json", "vocab.json", "merges.txt", "tokenizer.json"]:
        copy_file_if_exists(src_path, output_dir, filename)

    print("\nCheckpoint saved successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Save Qwen3VL-DINOv3 mixed model as Hugging Face format checkpoint"
    )
    parser.add_argument("--qwen3vl_model_path", type=str, default="/mnt/data/models/Qwen/Qwen3-0.6B")
    parser.add_argument("--dinov3_model_path", type=str, default="/mnt/data/models/facebook/dinov3-vits16plus-pretrain-lvd1689m")
    parser.add_argument("--output_dir", type=str, default="/mnt/data/models/qwen3VL-DINOv3-Flex-0.6B-Instruct-700")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--num_cameras", type=int, default=6)
    parser.add_argument("--num_timesteps", type=int, default=2)
    parser.add_argument("--num_scene_tokens", type=int, default=900)
    parser.add_argument("--flex_hidden_dim", type=int, default=768)

    args = parser.parse_args()
    
    save_qwen3vl_dinov3vit_checkpoint(
        qwen3vl_model_path=args.qwen3vl_model_path,
        dinov3_model_path=args.dinov3_model_path,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint_path,
        num_cameras=args.num_cameras,
        num_timesteps=args.num_timesteps,
        num_scene_tokens=args.num_scene_tokens,
        flex_hidden_dim=args.flex_hidden_dim
    )


if __name__ == "__main__":
    main()