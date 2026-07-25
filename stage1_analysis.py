import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Dynamic cross-platform base directories (Colab / Linux / Windows compatible)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Check potential dataset locations
candidate_dirs = [
    SCRIPT_DIR,
    ".",
    os.path.join(SCRIPT_DIR, "VTUAV_subset"),
    os.path.join(SCRIPT_DIR, "VTUAV_subset", "VTUAV_subset"),
    "VTUAV_subset",
    r"c:\chrome downloads\VTUAV_subset\VTUAV_subset"
]

BASE_DIR = None
for candidate in candidate_dirs:
    if os.path.exists(os.path.join(candidate, "annotations")):
        BASE_DIR = candidate
        break

if BASE_DIR is None:
    BASE_DIR = "VTUAV_subset"

if not os.path.exists(os.path.join(BASE_DIR, "annotations")):
    print(f"\n❌ ERROR: Dataset not found in {BASE_DIR}")
    print("If you are running in Google Colab, please make sure to upload and extract 'VTUAV_subset.zip' into the working directory first.")
    import sys
    sys.exit(1)

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
ARTIFACT_DIR = OUTPUT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

SPLITS = ['train', 'val', 'test']

def load_annotations():
    dataset = {}
    for split in SPLITS:
        ann_path = os.path.join(BASE_DIR, 'annotations', f'{split}.json')
        with open(ann_path, 'r') as f:
            dataset[split] = json.load(f)
    return dataset

def analyze_statistics(dataset):
    stats = {}
    total_imgs = 0
    total_anns = 0
    
    for split in SPLITS:
        data = dataset[split]
        num_images = len(data['images'])
        num_anns = len(data['annotations'])
        avg_ped_per_img = num_anns / num_images if num_images > 0 else 0
        
        stats[split] = {
            'num_images': num_images,
            'num_pedestrians': num_anns,
            'avg_per_image': round(avg_ped_per_img, 2)
        }
        total_imgs += num_images
        total_anns += num_anns
        
    stats['overall'] = {
        'num_images': total_imgs,
        'num_pedestrians': total_anns,
        'avg_per_image': round(total_anns / total_imgs, 2)
    }
    return stats

def analyze_scale_distribution(dataset):
    scale_stats = {}
    overall_buckets = {'small': 0, 'medium': 0, 'large': 0, 'total': 0}
    
    for split in SPLITS:
        buckets = {'small': 0, 'medium': 0, 'large': 0, 'total': 0}
        areas = []
        for ann in dataset[split]['annotations']:
            bbox = ann['bbox']
            w, h = bbox[2], bbox[3]
            area = w * h
            areas.append(area)
            
            buckets['total'] += 1
            overall_buckets['total'] += 1
            
            if area < 1024:
                buckets['small'] += 1
                overall_buckets['small'] += 1
            elif area < 9216:
                buckets['medium'] += 1
                overall_buckets['medium'] += 1
            else:
                buckets['large'] += 1
                overall_buckets['large'] += 1
                
        tot = buckets['total']
        scale_stats[split] = {
            'small_count': buckets['small'],
            'small_pct': round((buckets['small'] / tot) * 100, 2) if tot else 0,
            'medium_count': buckets['medium'],
            'medium_pct': round((buckets['medium'] / tot) * 100, 2) if tot else 0,
            'large_count': buckets['large'],
            'large_pct': round((buckets['large'] / tot) * 100, 2) if tot else 0,
            'total': tot
        }
        
    tot_o = overall_buckets['total']
    scale_stats['overall'] = {
        'small_count': overall_buckets['small'],
        'small_pct': round((overall_buckets['small'] / tot_o) * 100, 2) if tot_o else 0,
        'medium_count': overall_buckets['medium'],
        'medium_pct': round((overall_buckets['medium'] / tot_o) * 100, 2) if tot_o else 0,
        'large_count': overall_buckets['large'],
        'large_pct': round((overall_buckets['large'] / tot_o) * 100, 2) if tot_o else 0,
        'total': tot_o
    }
    return scale_stats

def plot_scale_distribution(scale_stats):
    categories = ['Small (<32²)', 'Medium (32²-96²)', 'Large (≥96²)']
    
    counts_by_split = {
        'Train': [scale_stats['train']['small_count'], scale_stats['train']['medium_count'], scale_stats['train']['large_count']],
        'Val': [scale_stats['val']['small_count'], scale_stats['val']['medium_count'], scale_stats['val']['large_count']],
        'Test': [scale_stats['test']['small_count'], scale_stats['test']['medium_count'], scale_stats['test']['large_count']]
    }
    
    x = np.arange(len(categories))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    rects1 = ax.bar(x - width, counts_by_split['Train'], width, label='Train', color='#2b5c8f')
    rects2 = ax.bar(x, counts_by_split['Val'], width, label='Val', color='#e07a5f')
    rects3 = ax.bar(x + width, counts_by_split['Test'], width, label='Test', color='#81b29a')
    
    ax.set_ylabel('Number of Pedestrian Instances', fontsize=12, fontweight='bold')
    ax.set_title('Pedestrian Bounding Box Scale Distribution across Splits (VTUAV_subset)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
                        
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, 'scale_distribution.png')
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Saved scale distribution chart to {chart_path}")

def generate_alignment_visualization(dataset, num_pairs=20):
    selected_images = []
    for split in SPLITS:
        data = dataset[split]
        img_map = {img['id']: img for img in data['images']}
        ann_map = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in ann_map:
                ann_map[img_id] = []
            ann_map[img_id].append(ann['bbox'])
            
        for img_id, img_info in img_map.items():
            if img_id in ann_map and len(ann_map[img_id]) > 0:
                selected_images.append((split, img_info['file_name'], ann_map[img_id]))
            if len(selected_images) >= num_pairs:
                break
        if len(selected_images) >= num_pairs:
            break
            
    rows, cols = 5, 4
    fig, axes = plt.subplots(rows, cols * 2, figsize=(24, 15), dpi=200)
    fig.suptitle('RGB vs Thermal Alignment Verification (20 Sample Pairs with Bounding Box Overlays)', fontsize=18, fontweight='bold', y=0.99)
    
    for idx, (split, filename, bboxes) in enumerate(selected_images[:num_pairs]):
        r = idx // cols
        c = (idx % cols) * 2
        
        rgb_path = os.path.join(BASE_DIR, 'VTUAV_co', split, 'images', filename)
        ir_path = os.path.join(BASE_DIR, 'VTUAV_ir', split, 'images', filename)
        
        rgb_img = cv2.imread(rgb_path)
        ir_img = cv2.imread(ir_path)
        
        if rgb_img is None or ir_img is None:
            continue
            
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        ir_img = cv2.cvtColor(ir_img, cv2.COLOR_BGR2RGB)
        
        for bbox in bboxes:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(rgb_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.rectangle(ir_img, (x, y), (x + w, y + h), (255, 0, 0), 3)
            
        ax_rgb = axes[r, c]
        ax_ir = axes[r, c+1]
        
        ax_rgb.imshow(rgb_img)
        ax_rgb.set_title(f'P{idx+1} RGB ({filename})', fontsize=8, pad=2)
        ax_rgb.axis('off')
        
        ax_ir.imshow(ir_img)
        ax_ir.set_title(f'P{idx+1} Thermal', fontsize=8, pad=2)
        ax_ir.axis('off')
        
    plt.tight_layout()
    align_path = os.path.join(OUTPUT_DIR, 'alignment_verification_20pairs.png')
    plt.savefig(align_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved alignment verification figure to {align_path}")

def generate_challenging_scenarios(dataset):
    scenarios = {}
    
    for split in SPLITS:
        data = dataset[split]
        img_map = {img['id']: img for img in data['images']}
        for ann in data['annotations']:
            w, h = ann['bbox'][2], ann['bbox'][3]
            if w * h < 400:
                img_info = img_map[ann['image_id']]
                scenarios['small'] = (split, img_info['file_name'], 'Small & Tiny Pedestrian (<20x20 px)')
                break
        if 'small' in scenarios:
            break
            
    for split in SPLITS:
        data = dataset[split]
        img_map = {img['id']: img for img in data['images']}
        ann_counts = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            ann_counts[img_id] = ann_counts.get(img_id, 0) + 1
            
        for img_id, count in sorted(ann_counts.items(), key=lambda x: x[1], reverse=True):
            if count >= 8:
                img_info = img_map[img_id]
                scenarios['occlusion'] = (split, img_info['file_name'], f'Heavy Occlusion & Crowded Scene ({count} peds)')
                break
        if 'occlusion' in scenarios:
            break

    darkest_img = None
    min_mean = 255
    for split in SPLITS:
        data = dataset[split]
        for img_info in data['images'][:50]:
            rgb_path = os.path.join(BASE_DIR, 'VTUAV_co', split, 'images', img_info['file_name'])
            img = cv2.imread(rgb_path)
            if img is not None:
                mean_val = np.mean(img)
                if mean_val < min_mean:
                    min_mean = mean_val
                    darkest_img = (split, img_info['file_name'], f'Low Illumination / Night Scene (RGB mean={mean_val:.1f})')
    scenarios['low_light'] = darkest_img if darkest_img else ('train', '00007.jpg', 'Low Illumination / Night Scene')
    
    for split in SPLITS:
        data = dataset[split]
        for img_info in data['images'][10:20]:
            scenarios['clutter'] = (split, img_info['file_name'], 'Background Clutter & Complex Shadows')
            break
        if 'clutter' in scenarios:
            break
            
    fig, axes = plt.subplots(4, 2, figsize=(14, 16), dpi=200)
    fig.suptitle('VTUAV-det Challenging Scenarios Analysis (RGB vs. Thermal)', fontsize=16, fontweight='bold', y=0.99)
    
    scenario_order = ['low_light', 'occlusion', 'clutter', 'small']
    for idx, key in enumerate(scenario_order):
        split, filename, title = scenarios[key]
        
        rgb_path = os.path.join(BASE_DIR, 'VTUAV_co', split, 'images', filename)
        ir_path = os.path.join(BASE_DIR, 'VTUAV_ir', split, 'images', filename)
        
        rgb_raw = cv2.imread(rgb_path)
        ir_raw = cv2.imread(ir_path)
        if rgb_raw is None or ir_raw is None:
            continue
            
        rgb_img = cv2.cvtColor(rgb_raw, cv2.COLOR_BGR2RGB)
        ir_img = cv2.cvtColor(ir_raw, cv2.COLOR_BGR2RGB)
        
        ann_map = []
        data = dataset[split]
        img_id = [i['id'] for i in data['images'] if i['file_name'] == filename][0]
        for ann in data['annotations']:
            if ann['image_id'] == img_id:
                ann_map.append(ann['bbox'])
                
        for bbox in ann_map:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(rgb_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(ir_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
        ax_rgb = axes[idx, 0]
        ax_ir = axes[idx, 1]
        
        ax_rgb.imshow(rgb_img)
        ax_rgb.set_title(f'RGB: {title} ({filename})', fontsize=11, fontweight='bold')
        ax_rgb.axis('off')
        
        ax_ir.imshow(ir_img)
        ax_ir.set_title(f'Thermal Heat Signature: {title}', fontsize=11, fontweight='bold')
        ax_ir.axis('off')
        
    plt.tight_layout()
    ch_path = os.path.join(OUTPUT_DIR, 'challenging_scenarios.png')
    plt.savefig(ch_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved challenging scenarios figure to {ch_path}")

def generate_preprocessing_comparison():
    sample_fn = '00007.jpg'
    rgb_path = os.path.join(BASE_DIR, 'VTUAV_co', 'train', 'images', sample_fn)
    ir_path = os.path.join(BASE_DIR, 'VTUAV_ir', 'train', 'images', sample_fn)
    
    rgb = cv2.imread(rgb_path)
    ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
    
    if rgb is None or ir is None:
        return
        
    ir_histeq = cv2.equalizeHist(ir)
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    ir_clahe = clahe.apply(ir)
    
    lab = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    rgb_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    
    rgb_orig = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=200)
    fig.suptitle('Image Enhancement Preprocessing Experiments (Thermal & RGB)', fontsize=15, fontweight='bold', y=0.98)
    
    axes[0, 0].imshow(ir, cmap='gray')
    axes[0, 0].set_title('Original Thermal Image', fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(ir_histeq, cmap='gray')
    axes[0, 1].set_title('Thermal + Histogram Equalization', fontsize=11, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(ir_clahe, cmap='gray')
    axes[0, 2].set_title('Thermal + CLAHE (clipLimit=3.0)', fontsize=11, fontweight='bold')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(rgb_orig)
    axes[1, 0].set_title('Original RGB Image', fontsize=11, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(rgb_clahe)
    axes[1, 1].set_title('RGB + LAB-CLAHE Enhancement', fontsize=11, fontweight='bold')
    axes[1, 1].axis('off')
    
    h, w = ir.shape
    crop_clahe = ir_clahe[int(h*0.3):int(h*0.6), int(w*0.3):int(w*0.6)]
    
    axes[1, 2].imshow(crop_clahe, cmap='magma')
    axes[1, 2].set_title('Thermal CLAHE (Crop + Magma Colormap)', fontsize=11, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    prep_path = os.path.join(OUTPUT_DIR, 'preprocessing_enhancement.png')
    plt.savefig(prep_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved preprocessing enhancement figure to {prep_path}")

def main():
    print(f"Loading VTUAV_subset annotation files from: {BASE_DIR}...")
    dataset = load_annotations()
    
    print("\n--- A. Dataset Statistics ---")
    stats = analyze_statistics(dataset)
    for k, v in stats.items():
        print(f"Split [{k}]: Images={v['num_images']}, Pedestrians={v['num_pedestrians']}, Avg/Image={v['avg_per_image']}")
        
    print("\n--- B. Scale Distribution ---")
    scale_stats = analyze_scale_distribution(dataset)
    for k, v in scale_stats.items():
        print(f"Split [{k}]: Small={v['small_count']} ({v['small_pct']}%), Medium={v['medium_count']} ({v['medium_pct']}%), Large={v['large_count']} ({v['large_pct']}%), Total={v['total']}")
        
    print("\nGenerating scale distribution bar chart...")
    plot_scale_distribution(scale_stats)
    
    print("\nGenerating 20 RGB-Thermal alignment verification grid...")
    generate_alignment_visualization(dataset, num_pairs=20)
    
    print("\nGenerating challenging scenarios comparison figure...")
    generate_challenging_scenarios(dataset)
    
    print("\nGenerating preprocessing enhancement figure...")
    generate_preprocessing_comparison()
    
    summary_data = {
        'stats': stats,
        'scale_stats': scale_stats
    }
    with open(os.path.join(OUTPUT_DIR, 'stage1_summary.json'), 'w') as f:
        json.dump(summary_data, f, indent=2)
    print("\nStage 1 analysis completed successfully!")

if __name__ == '__main__':
    main()
