import torch
import numpy as np
import os

USE_AMAZON = True
AMAZON_DATASET = 'last-fm'
# AMAZON_DATASET = 'yelp2018'
# AMAZON_DATASET = 'last-fm'

_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)

if os.path.exists(os.path.join(_current_dir, 'data')):
    DATA_ROOT = './data'
elif os.path.exists(os.path.join(_parent_dir, 'data')):
    DATA_ROOT = '../data'
else:
    DATA_ROOT = './data'

AMAZON_DATASET_PATH = f'{DATA_ROOT}/{AMAZON_DATASET}'
MACHINING_DATA_DIR = f'{DATA_ROOT}/machining_process'
DATASET_PATH = AMAZON_DATASET_PATH
DATASET_NAME = 'Amazon-Book'

if not os.path.exists(os.path.join(_current_dir, 'checkpoints')) and \
   os.path.exists(os.path.join(_parent_dir, 'checkpoints')):
    CHECKPOINT_DIR = os.path.join(_parent_dir, 'checkpoints')
    RESULT_DIR = os.path.join(_parent_dir, 'results')
    LOG_DIR = os.path.join(_parent_dir, 'logs')
else:
    CHECKPOINT_DIR = './checkpoints'
    RESULT_DIR = './results'
    LOG_DIR = './logs'

for d in [CHECKPOINT_DIR, RESULT_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = torch.device('cpu')
    print("⚠️ Using CPU")

EMBED_DIM = 64
BATCH_SIZE = 1024
KG_BATCH_SIZE = 1024
TOP_K_LIST = [1, 3, 5, 10, 20]
NUM_WORKERS = 0
PIN_MEMORY = True if torch.cuda.is_available() else False
USE_AMP = True
L2_REG = 1e-4

PCAGAT_LAYERS = 3
PCAGAT_HEADS = 1
PCAGAT_DROPOUT = 0.2
PCAGAT_NEGATIVE_SLOPE = 0.2

USE_RESIDUAL = True
USE_CROSS_LAYER_RESIDUAL = False

LAYER_AGG_MODE = 'concat'
USE_SIMPLE_ATTENTION = False
USE_BI_INTERACTION = False

EARLY_STOP_METRIC = 'R@20'

AUTO_DISABLE_CONSTRAINT = True

CONSTRAINT_LAMBDA_INIT = 0.5
CONSTRAINT_TYPES = [
    'material_operation',
    'precision_operation',
    'operation_sequence',
    'feature_operation',
]
USE_LEARNABLE_LAMBDA = True

USE_CONSTRAINT_GATE = False
CONSTRAINT_GATE_HIDDEN = None

USE_CONSTRAINT_CONTRASTIVE = False
CONSTRAINT_CONTRASTIVE_WEIGHT = 0.0
CONSTRAINT_CONTRASTIVE_MARGIN = 0.5

USE_CONSTRAINT_ALIGNMENT = False
CONSTRAINT_ALIGNMENT_WEIGHT = 0.0
CONSTRAINT_ALIGNMENT_SAMPLE = 2048

USE_BOUNDED_LAMBDA = True
LAMBDA_GLOBAL_SCALE = 2.0
LAMBDA_MIN = 0.10
LAMBDA_GLOBAL_MIN = 0.20

USE_HARD_NEG_SAMPLING = False
HARD_NEG_RATIO = 0.0

KG_EMBED_METHOD = 'TransE'
KG_MARGIN = 1.0
TRANSE_NORM = 1

PCAGAT_EPOCHS = 300
PCAGAT_LR = 0.0005
PCAGAT_KG_LR = 0.001
PCAGAT_CONSTRAINT_LR_MULT = 10
PCAGAT_PATIENCE = 15
PCAGAT_MIN_EPOCHS = 20
PCAGAT_EVAL_INTERVAL = 5
PCAGAT_KG_WEIGHT = 0.1
PCAGAT_L2_WEIGHT = 1e-4

USE_PRETRAIN = True
PRETRAIN_EPOCHS = 400
PRETRAIN_LR = 0.001
PRETRAIN_PATIENCE = 15

EXPLAIN_TOP_K_PATHS = 3
EXPLAIN_MAX_HOP = 3
EXPLAIN_ATTENTION_THRESHOLD = 0.05

PCAGAT_CHECKPOINT = os.path.join(CHECKPOINT_DIR, 'pcagat_best.pth')
PRETRAIN_CHECKPOINT = os.path.join(CHECKPOINT_DIR, 'pcagat_pretrain_bprmf.pth')

ABLATION_CONFIGS = {
    'full': {
        'desc': 'PCA-GAT (Ours)',
        'use_constraint': True,
        'use_gate': True,
        'use_contrastive': False,
        'use_hard_neg': False,
        'use_kg': True,
        'n_layers': PCAGAT_LAYERS,
        'n_heads': 1,
    },
    'wo_gate': {
        'desc': 'w/o Constraint Gate',
        'use_constraint': True,
        'use_gate': False,
        'use_contrastive': False,
        'use_hard_neg': False,
        'use_kg': True,
        'n_layers': PCAGAT_LAYERS,
        'n_heads': 1,
    },
    'wo_constraint': {
        'desc': 'w/o Process Constraint',
        'use_constraint': False,
        'use_gate': False,
        'use_contrastive': False,
        'use_hard_neg': False,
        'use_kg': True,
        'n_layers': PCAGAT_LAYERS,
        'n_heads': 1,
    },
    'wo_kg': {
        'desc': 'w/o Knowledge Graph',
        'use_constraint': False,
        'use_gate': False,
        'use_contrastive': False,
        'use_hard_neg': False,
        'use_kg': False,
        'n_layers': PCAGAT_LAYERS,
        'n_heads': 1,
    },
    'single_layer': {
        'desc': 'Single Layer (L=1)',
        'use_constraint': True,
        'use_gate': True,
        'use_contrastive': False,
        'use_hard_neg': False,
        'use_kg': True,
        'n_layers': 1,
        'n_heads': 1,
    },
    'multi_head': {
        'desc': 'Multi-Head (4 heads)',
        'use_constraint': True,
        'use_gate': True,
        'use_contrastive': False,
        'use_hard_neg': False,
        'use_kg': True,
        'n_layers': PCAGAT_LAYERS,
        'n_heads': 4,
    },
}

SEED = 1024
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

MAX_EVAL_USERS = None

print("\n" + "=" * 70)
print("   📊 PCA-GAT V8.0 (冲击0.148)")
print("=" * 70)
print(f"   Dataset: {DATASET_NAME}")
print(f"   Device: {DEVICE}")
print(f"   ★ Embed Dim: {EMBED_DIM}")
print(f"   ★ Layers: {PCAGAT_LAYERS}, Agg: {LAYER_AGG_MODE}")
print(f"   ★ Dropout: {PCAGAT_DROPOUT}")
print(f"   ★ Early Stop: {EARLY_STOP_METRIC}")
print(f"   ★ LR: {PCAGAT_LR}, Patience: {PCAGAT_PATIENCE}")
print(f"   ★ L2_WEIGHT: {PCAGAT_L2_WEIGHT}")
print(f"   ★ Bi-Interaction: {USE_BI_INTERACTION}")
print(f"   🔘 Residual Eq.9: {USE_RESIDUAL}  |  Eq.10: {USE_CROSS_LAYER_RESIDUAL}")
print("=" * 70 + "\n")