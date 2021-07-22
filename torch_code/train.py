import torch
import numpy as np
from models.models import Models
from utils.parser import setup_parser
from pathlib import Path
from torchsummary import summary
import json
from trainer import Trainer
from utils.plots import plot_hist2d
import os


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    # set parameters / parse arguments
    parser = setup_parser()
    args, unknown = parser.parse_known_args()

    if args.loss_key in ['gaussian_nll', 'laplacian_nll']:
        args.num_outputs = 2

    # log train/val metrics in tensorboard
    tensorboard_log_dir = Path(args.out_dir)/'log'

    # Setup model
    model_chooser = Models()
    model = model_chooser(args.model_name)(num_outputs=args.num_outputs)
    model.to(device)
    summary(model, input_size=(1, args.sample_length))

    # Setup Trainer
    trainer = Trainer(model=model, log_dir=tensorboard_log_dir, args=args)

    print('TRAIN: ', len(trainer.ds_train))
    print('VAL:   ', len(trainer.ds_val))
    print('TEST:  ', len(trainer.ds_test))

    # train
    if os.path.exists(Path(args.out_dir) / 'weights_last_epoch.pt') or args.skip_training:
        print("MODEL WAS ALREADY TRAINED. SKIP TRAINING! (because the file 'weights_last_epoch.pt' exists already)")
    else:
        trainer.train()
        
    # --- test ---
    if os.path.exists(Path(args.out_dir) / 'confusion.png'):
        print("MODEL WAS ALREADY TESTED. SKIP TESTING! (because the file 'confusion.png' exists already)")
    else:
        test_metrics, test_dict, test_metric_string = trainer.test()

        # save results
        with open(Path(args.out_dir) / 'results.txt', 'w') as f:
            f.write(test_metric_string)

        with open(Path(args.out_dir) / 'test_results.json', 'w') as f:
            json.dump(test_metrics, f)

        for key in test_dict.keys():
            np.save(file=Path(args.out_dir) / '{}.npy'.format(key), arr=test_dict[key])
        np.save(file=Path(args.out_dir) / 'test_indices.npy', arr=trainer.test_indices)

        # plot confusion ground truth vs. prediction
        plot_hist2d(x=test_dict['targets'], y=test_dict['predictions'], ma=args.max_gt, step=args.max_gt / 10,
                    out_dir=args.out_dir, figsize=(8, 6), xlabel='Ground truth [m]', ylabel='Prediction [m]')



