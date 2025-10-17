from ase.io import read as ase_read
from pymatgen.io.ase import AseAtomsAdaptor

import torch.nn as nn
from torch import tensor

from core.md_model import BackboneInterface
import os


class MachineLearningForce(BackboneInterface, nn.Module):

    def __init__(self,
                 molecular,
                 aimd_pos_file: str,
                 aimd_force_file: str,
                 mlp_model_name=None,
                 mlps_finetune_params=None,
                 mlps_model_path=None,
                 ):
        super(MachineLearningForce, self).__init__()
        self.molecular = molecular
        box_size = self.molecular.box_length
        if aimd_force_file.endswith(".xyz") and aimd_pos_file.endswith(".xyz"):
            self.aimd_pos_file = aimd_pos_file
            self.aimd_force_file = aimd_force_file
        else:
            raise ValueError('The input file should be in xyz format')
        self.cell_param = [box_size.cpu(), box_size.cpu(), box_size.cpu()]
        self.mlp_model_name = mlp_model_name
        self.device = self.molecular.device
        self.model = None
        if mlps_model_path:
            self.mlps_model_path = mlps_model_path
            self.load_model_from_path()
        else:
            self.mlps_finetune_params = mlps_finetune_params
            self.mlp_flow()
            self._load_model()

    def forward(self, atom_coordinates=None, cutoff=3, min_distance=1):

        if self.mlp_model_name == 'chgnet':
            structure = self.molecular.to_pymatgen_structure()
            pred = self.model.predict_structure(structure, task='ef')
            total_energy = tensor(pred['e'], device=self.device)
            atom_force = tensor(pred['f'], device=self.device)
        else:
            total_energy = 0
            atom_force = 0

        return {'energy': total_energy, 'forces': atom_force}

    def load_model_from_path(self):
        if self.mlp_model_name == 'chgnet':
            from chgnet.model import CHGNet
            self.model = CHGNet.from_file(self.mlps_model_path).to(self.device)
        else:
            raise NotImplementedError('Please provide a model name for the MLP model')

    def _load_model(self):
        if self.mlp_model_name == 'chgnet':
            root_dir = os.path.dirname(os.path.abspath(__file__))
            model_folder = os.path.join(root_dir, 'model')
            files = sorted(os.listdir(model_folder))
            if files:
                model_path = os.path.join(model_folder, files[0])
                from chgnet.model import CHGNet
                self.model = CHGNet.from_file(model_path).to(self.device)
            else:
                raise FileNotFoundError("No model found in 'model' folder")
        else:
            self.model = None

    def mlp_flow(self):
        if self.mlp_model_name is None:
            raise NotImplementedError('Please provide a model name for the MLP model')
        else:
            from chgnet.model import CHGNet
            self.finetune_large_mlp()

    def convert_aimd_to_dataset(self):
        if self.mlp_model_name == 'chgnet':
            position_frames = ase_read(self.aimd_pos_file, index=":")
            force_frames = ase_read(self.aimd_force_file, index=":")
            structures = []
            forces = []
            energies = []
            for pos_atoms, force_atoms in zip(position_frames, force_frames):
                pos_atoms.set_cell(self.cell_param)
                pos_atoms.set_pbc(True)
                structure = AseAtomsAdaptor().get_structure(pos_atoms)
                structures.append(structure)

                force_atom = force_atoms.positions
                forces.append(force_atom)

                energy = pos_atoms.info['E']
                energies.append(energy)
            dataset_dict = {"structures": structures, "forces": forces, "energies": energies}
            return dataset_dict

    def finetune_large_mlp(self):
        if self.mlp_model_name == 'chgnet':
            dataset_dict = self.convert_aimd_to_dataset()
            from chgnet.data.dataset import StructureData, get_train_val_test_loader
            dataset = StructureData(
                structures=dataset_dict['structures'],
                forces=dataset_dict['forces'],
                energies=dataset_dict['energies']
            )
            train_loader, val_loader, test_loader = get_train_val_test_loader(dataset, batch_size=8, train_ratio=0.9,
                                                                              val_ratio=0.05)
            from chgnet.model import CHGNet
            from chgnet.trainer import Trainer
            origin_chgnet = CHGNet()
            self.mlps_finetune_params['targets']
            for layer in [
                origin_chgnet.atom_embedding,
                origin_chgnet.bond_embedding,
                origin_chgnet.angle_embedding,
                origin_chgnet.bond_basis_expansion,
                origin_chgnet.angle_basis_expansion,
                origin_chgnet.atom_conv_layers[:-1],
                origin_chgnet.bond_conv_layers,
                origin_chgnet.angle_layers
            ]:
                for param in layer.parameters():
                    param.requires_grad = False
            if self.mlps_finetune_params:
                trainer = Trainer(
                    model=origin_chgnet,
                    targets=self.mlps_finetune_params['targets'],
                    optimizer=self.mlps_finetune_params['optimizer'],
                    scheduler=self.mlps_finetune_params['scheduler'],
                    criterion=self.mlps_finetune_params['criterion'],
                    epochs=self.mlps_finetune_params['epochs'],
                    learning_rate=self.mlps_finetune_params['learning_rate'],
                    use_device=self.device,
                    print_freq=self.mlps_finetune_params['print_freq']
                )
            else:
                trainer = Trainer(
                    model=origin_chgnet,
                    targets='ef',
                    optimizer='Adam',
                    scheduler='CosLR',
                    criterion='MSE',
                    epochs=10,
                    learning_rate=0.002,
                    use_device=self.device,
                    print_freq=6
                )
            model_folder = None
            try:
                import os
                root_dir = os.path.dirname(os.path.abspath(__file__))
                model_folder = os.path.join(root_dir, 'model')
                os.makedirs(model_folder, exist_ok=True)
            except OSError as e:
                print(e)
            if model_folder is not None:
                trainer.train(train_loader, val_loader, test_loader, save_dir=model_folder)
            else:
                trainer.train(train_loader, val_loader, test_loader)
                model = trainer.best_model
                return model
