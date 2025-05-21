from easydict import EasyDict as edict

config = edict()
config.dataset = "webface" # training dataset
config.embedding_size = 512 # embedding size of model
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 256 # batch size per GPU
config.lr = 0.1
config.output = "../../../../data/mcaldeir/data_pruning" # train model output folder
config.global_step=0 # step to resume
config.s=64.0
config.m=0.35 # margin for CosFace: 0.35
config.std=0.05


config.loss="CosFace" # options: CosFace, ArcFace, CurricularFace, AdaFace

if (config.loss=="ElasticArcFacePlus"):
    config.s = 64.0
    config.m = 0.50
    config.std = 0.0175
elif (config.loss=="ElasticArcFace"):
    config.s = 64.0
    config.m = 0.50
    config.std = 0.05
elif (config.loss=="ArcFace" or config.loss=="CurricularFace"):
    config.m=0.5

if (config.loss=="ElasticCosFacePlus"):
    config.s = 64.0
    config.m = 0.35
    config.std = 0.02
elif (config.loss=="ElasticCosFace"):
    config.s = 64.0
    config.m = 0.35
    config.std = 0.05
elif (config.loss=="CosFace"):
    config.m=0.35


# type of network to train [iresnet100 | iresnet50]
config.network="iresnet50" # "iresnet50", "iresnet34", "mfn"
config.SE=False # SEModule

config.original_net_loss="iresnet50_ArcFace"
config.is_original_train=False
config.pruned_net_loss=config.network+"_"+config.loss

# ====================================================================================================================
# CORESET DETERMINATION CONFIGS

config.eval_epoch=34

config.fraction=0.25
config.window=10
config.min_samples_per_id=5
config.threshold=0.00003
config.use_all_epochs=False

config.coreset_input=config.output
config.coreset_output=config.output+'/coresets'

config.coreset_order='local' # options: 'local'
config.coreset_method='eval_simprobs' # options: 'rand', 'dynunc', 'eval_simprobs', 'eval_simprobs_clean'

# ====================================================================================================================

if config.dataset == "emoreIresNet":
    config.rec = "/data/psiebke/faces_emore"
    config.num_classes = 85742
    config.num_image = 5822653
    config.num_epoch =  26
    config.warmup_epoch = -1
    config.val_targets =  ["lfw", "cfp_fp", "cfp_ff", "agedb_30", "calfw", "cplfw"]
    config.eval_step=5686
    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < -1 else 0.1 ** len(
            [m for m in [8, 14,20,25] if m - 1 <= epoch])  # [m for m in [8, 14,20,25] if m - 1 <= epoch])
    config.lr_func = lr_step_func

elif config.dataset == "webface":
    config.rec = "/data/Biometrics/database/faces_webface_112x112"

    config.num_classes_original = 10572

    if config.coreset_method == 'eval_simprobs_clean':
        config.num_classes = 10562
    else:
        config.num_classes = config.num_classes_original

    config.num_image = 490623
    config.num_epoch = 34   #  [22, 30, 35]
    config.warmup_epoch = -1
    config.val_targets = ["lfw", "cfp_fp", "cfp_ff", "agedb_30", "calfw", "cplfw"]
    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < config.warmup_epoch else 0.1 ** len(
            [m for m in [20, 28, 32] if m - 1 <= epoch])
    config.lr_func = lr_step_func