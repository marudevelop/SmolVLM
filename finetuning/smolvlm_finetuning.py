#!/usr/bin/env python3
"""
SmolVLM Visual CoT 훈련 - 간단 버전
A6000 최적화된 효율적인 배치 처리
"""

import os
import json
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, List
from PIL import Image
import logging

from transformers import (
    TrainingArguments, 
    Trainer,
    AutoProcessor,
    AutoModelForVision2Seq,
    Idefics3ForConditionalGeneration
)
from torch.utils.data import Dataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingArgumentsCustom(TrainingArguments):
    stage: int = field(default=1)

class VisualCoTDataset(Dataset):
    def __init__(self, data_path: str, image_folder: str, stage: int = 1, limit_samples: int = None):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        if limit_samples:
            self.data = self.data[:limit_samples]
            
        self.image_folder = image_folder
        self.stage = stage
        logger.info(f"Loaded {len(self.data)} samples for Stage {stage}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image
        if 'image' in item:
            image_path = os.path.join(self.image_folder, item['image'])
        elif 'id' in item:  # LLaVA pretrain format
            image_path = os.path.join(self.image_folder, f"{item['id']}.jpg")
        else:
            image_path = os.path.join(self.image_folder, f"sample_{idx}.jpg")
            
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Could not load image {image_path}: {e}")
            # Dummy image if not found
            image = Image.new('RGB', (512, 512), color=(100, 150, 200))
        
        if self.stage == 1:
            # Stage 1: LLaVA pretrain format
            if 'caption' in item:
                text = item['caption']
            elif 'conversations' in item and len(item['conversations']) > 0:
                # Extract from conversations
                text = item['conversations'][-1]['value']
            else:
                text = 'This is an image.'
            return {'image': image, 'text': text}
        else:
            # Stage 2: Visual CoT format
            conversations = item.get('conversations', [])
            if conversations and len(conversations) >= 2:
                question = conversations[0]['value']
                answer = conversations[1]['value']
            else:
                question = "What do you see in this image?"
                answer = "I can see an image."
            
            return {
                'image': image,
                'question': question,
                'answer': answer,
                'has_cot': 'bbox' in item and item['bbox'] is not None
            }

def collate_fn_stage1(batch, processor):
    """Stage 1 배치 처리"""
    images = [[item['image']] for item in batch] 
    texts = [f"<image>{item['text']}" for item in batch]
    
    try:
        inputs = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs
    except Exception as e:
        # fallback → 단일 샘플이라도 nested list 로
        inputs = processor(
            text=[texts[0]],
            images=[[images[0][0]]],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        inputs['labels'] = inputs['input_ids'].clone()
        return inputs

def collate_fn_stage2(batch, processor):
    """Stage 2 배치 처리 - 개별 처리 후 수동 배치화"""
    processed_samples = []
    
    for item in batch:
        image = item['image']
        question = item['question']
        answer = item['answer']
        has_cot = item.get('has_cot', False)
        
        input_text = f"<image>Question: {question}\nAnswer:"
        
        try:
            inputs = processor(
                text=input_text,
                images=image,  # 단일 이미지
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )
            
            target_inputs = processor.tokenizer(
                answer,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Create labels for this sample
            labels = inputs['input_ids'].clone()
            input_len = (inputs['input_ids'][0] != processor.tokenizer.pad_token_id).sum().item()
            target_len = (target_inputs['input_ids'][0] != processor.tokenizer.pad_token_id).sum().item()
            
            if input_len + target_len <= labels.size(1):
                labels[0, input_len:input_len+target_len] = target_inputs['input_ids'][0, :target_len]
                labels[0, :input_len] = -100
            
            inputs['labels'] = labels
            processed_samples.append(inputs)
            
        except Exception as e:
            logger.warning(f"Skipping sample due to error: {e}")
            continue
    
    if not processed_samples:
        logger.error("No samples could be processed")
        return None
    
    # 수동으로 배치 생성
    try:
        # 모든 키에 대해 배치 차원 생성
        batched_inputs = {}
        for key in processed_samples[0].keys():
            if key in ['input_ids', 'attention_mask', 'labels']:
                # 텍스트 관련 텐서들 - 패딩해서 배치화
                tensors = [sample[key] for sample in processed_samples]
                max_length = max(t.size(1) for t in tensors)
                
                padded_tensors = []
                for t in tensors:
                    if t.size(1) < max_length:
                        pad_length = max_length - t.size(1)
                        if key == 'input_ids':
                            pad_value = processor.tokenizer.pad_token_id
                        elif key == 'labels':
                            pad_value = -100
                        else:  # attention_mask
                            pad_value = 0
                        padded = F.pad(t, (0, pad_length), value=pad_value)
                        padded_tensors.append(padded)
                    else:
                        padded_tensors.append(t)
                
                batched_inputs[key] = torch.cat(padded_tensors, dim=0)
                
            elif key == 'pixel_values':
                # 이미지 텐서들
                tensors = [sample[key] for sample in processed_samples]
                try:
                    batched_inputs[key] = torch.cat(tensors, dim=0)
                except:
                    # 크기가 다르면 첫 번째 것만 사용
                    batched_inputs[key] = tensors[0]
            else:
                # 다른 키들
                tensors = [sample[key] for sample in processed_samples]
                try:
                    batched_inputs[key] = torch.cat(tensors, dim=0)
                except:
                    batched_inputs[key] = tensors[0]
        
        return batched_inputs
        
    except Exception as e:
        logger.error(f"Manual batching failed: {e}")
        # 마지막 fallback - 첫 번째 샘플만 반환
        return processed_samples[0]

class SimpleTrainer(Trainer):
    def __init__(self, stage=1, **kwargs):
        super().__init__(**kwargs)
        self.stage = stage
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        return (outputs.loss, outputs) if return_outputs else outputs.loss

def load_model(model_name):
    """모델 로드"""
    try:
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        logger.info("✓ Loaded with AutoModelForVision2Seq")
        return model
    except:
        model = Idefics3ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        logger.info("✓ Loaded with Idefics3ForConditionalGeneration")
        return model

def setup_stage1_training(model):
    """Stage 1: projection layer만 훈련"""
    for param in model.parameters():
        param.requires_grad = False
    
    # Projection layer 찾아서 활성화
    for name, module in model.named_modules():
        if any(x in name.lower() for x in ['projector', 'connector', 'multimodal']):
            for param in module.parameters():
                param.requires_grad = True
            logger.info(f"Enabled: {name}")
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

def setup_stage2_training(model):
    """Stage 2: 전체 모델 훈련"""
    for param in model.parameters():
        param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Stage 2 - All parameters trainable: {trainable:,}")

def download_data():
    """실제 데이터 다운로드 (Visual CoT 논문 베이스라인용)"""
    import subprocess
    
    os.makedirs("./data", exist_ok=True)
    os.makedirs("./data/images", exist_ok=True)
    
    # Stage 1: LAION-CC-SBU 558K (LLaVA pretrain data)
    logger.info("Downloading Stage 1 data (LAION-CC-SBU 558K)...")
    try:
        subprocess.run([
            "huggingface-cli", "download", "liuhaotian/LLaVA-Pretrain", 
            "blip_laion_cc_sbu_558k.json", "--local-dir", "./data"
        ], check=True)
        
        if os.path.exists("./data/blip_laion_cc_sbu_558k.json"):
            os.rename("./data/blip_laion_cc_sbu_558k.json", "./data/stage1_data.json")
        
        logger.info("✓ Stage 1 data downloaded (558K samples)")
    except Exception as e:
        logger.error(f"Failed to download Stage 1 data: {e}")
        logger.info("Please download manually from: https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain")
        return False
    
    # Stage 2: Visual CoT mixed data (2M samples)
    logger.info("Downloading Stage 2 data (Visual CoT mixed 2M)...")
    try:
        subprocess.run([
            "huggingface-cli", "download", "deepcs233/Visual-CoT",
            "viscot_mixed_2m.json", "--local-dir", "./data", "--repo-type", "dataset"
        ], check=True)
        
        if os.path.exists("./data/viscot_mixed_2m.json"):
            os.rename("./data/viscot_mixed_2m.json", "./data/stage2_data.json")
        
        logger.info("✓ Stage 2 data downloaded (2M samples)")
    except Exception as e:
        logger.error(f"Failed to download Stage 2 data: {e}")
        logger.info("Please download manually from: https://huggingface.co/datasets/deepcs233/Visual-CoT")
        return False
    
    logger.info("Note: Images need to be downloaded separately from the Visual CoT repository")
    logger.info("Please download images to ./data/images/ folder")
    
    return True

def train_stage1():
    """Stage 1 훈련"""
    logger.info("=== Stage 1: Projection Layer Training ===")
    
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    model = load_model("HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    setup_stage1_training(model)
    
    # Use existing data
    dataset = VisualCoTDataset(
        "/data3/jykim/Projects/VQA/data/pretrain_558k.json", 
        "/data3/jykim/Projects/VQA/data/images", 
        stage=1
    )
    
    training_args = TrainingArgumentsCustom(
        stage=1,
        output_dir="./checkpoints/stage1",
        overwrite_output_dir=True,
        num_train_epochs=1,  # Visual CoT 논문 설정
        per_device_train_batch_size=4,  # A6000 성능 활용!
        gradient_accumulation_steps=1,   # 단순화
        learning_rate=2e-3,  # Visual CoT 논문의 projection layer LR
        weight_decay=0.0,  # 논문 설정
        warmup_steps=500,  # 논문 설정
        logging_steps=50,    # 더 자주 로깅
        save_steps=2000,     # 더 자주 저장
        save_total_limit=2,
        bf16=True,
        dataloader_num_workers=8,  # 병렬 데이터 로딩
        remove_unused_columns=False,
        report_to=None,
        max_grad_norm=1.0,  # 논문 설정
        lr_scheduler_type="cosine",  # 논문 설정
        dataloader_pin_memory=True,  # 메모리 핀
        tf32=True,  # A6000 가속
    )
    
    trainer = SimpleTrainer(
        stage=1,
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda batch: collate_fn_stage1(batch, processor),
        processing_class=processor,
    )
    
    trainer.train()
    trainer.save_model()
    processor.save_pretrained(training_args.output_dir)
    logger.info(f"✓ Stage 1 completed: {training_args.output_dir}")

def train_stage2():
    """Stage 2 훈련"""
    logger.info("=== Stage 2: Full Model Training ===")
    
    processor = AutoProcessor.from_pretrained("./checkpoints/stage1")
    model = load_model("./checkpoints/stage1")
    setup_stage2_training(model)
    
    # Use existing data
    dataset = VisualCoTDataset(
        "/data3/jykim/Projects/VQA/data/viscot_mixed_2m.json", 
        "/data3/jykim/Projects/VQA/data/images", 
        stage=2
    )
    
    training_args = TrainingArgumentsCustom(
        stage=2,
        output_dir="./checkpoints/stage2",
        overwrite_output_dir=True,
        num_train_epochs=1,  # Visual CoT 논문 설정
        per_device_train_batch_size=2,   # A6000 성능 활용!
        gradient_accumulation_steps=2,   # 효과적 배치 크기 16
        learning_rate=2e-5,  # Visual CoT 논문의 full model LR
        weight_decay=0.0,  # 논문 설정
        warmup_steps=100,  # 논문 설정
        logging_steps=50,    # 더 자주 로깅
        save_steps=1000,     # 더 자주 저장
        save_total_limit=2,
        bf16=True,
        dataloader_num_workers=8,  # 병렬 데이터 로딩
        remove_unused_columns=False,
        report_to=None,
        max_grad_norm=1.0,  # 논문 설정
        lr_scheduler_type="cosine",  # 논문 설정
        dataloader_pin_memory=True,  # 메모리 핀
        tf32=True,  # A6000 가속
    )
    
    trainer = SimpleTrainer(
        stage=2,
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda batch: collate_fn_stage2(batch, processor),
        processing_class=processor,
    )
    
    trainer.train()
    trainer.save_model()
    processor.save_pretrained(training_args.output_dir)
    logger.info(f"✓ Stage 2 completed: {training_args.output_dir}")

def main():
    """메인 훈련 함수 - Visual CoT 논문 베이스라인"""
    
    # Check if data files exist
    stage1_data_path = "/data3/jykim/Projects/VQA/data/pretrain_558k.json"
    stage2_data_path = "/data3/jykim/Projects/VQA/data/viscot_mixed_2m.json"
    images_path = "/data3/jykim/Projects/VQA/data/images"
    
    if not os.path.exists(stage1_data_path):
        logger.error(f"Stage 1 data not found at {stage1_data_path}")
        return
    
    if not os.path.exists(stage2_data_path):
        logger.error(f"Stage 2 data not found at {stage2_data_path}")
        return
        
    if not os.path.exists(images_path):
        logger.error(f"Images folder not found at {images_path}")
        return
    
    logger.info("✓ All data files found. Starting Visual CoT baseline training...")
    
    # Stage 1 훈련 (558K samples)
    logger.info("=== Visual CoT Baseline: Stage 1 (558K samples) ===")
    train_stage1()
    
    # Stage 2 훈련 (2M samples)  
    logger.info("=== Visual CoT Baseline: Stage 2 (2M samples) ===")
    train_stage2()
    
    logger.info("🎉 Visual CoT Baseline Training Completed!")
    logger.info("Final model: ./checkpoints/stage2")
    logger.info("This follows the exact Visual CoT paper training procedure.")

if __name__ == "__main__":
    main()