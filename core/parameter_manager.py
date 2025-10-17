import torch
import itertools
from typing import List, Dict, Set, Any
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class ElementParameterManager:


    def __init__(self, element_list: List[str], parameter_dict: Dict[str, Dict[str, float]]):
       
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.element_list = element_list
        self.pair_list = ['-'.join(pair) for pair in itertools.product(element_list, repeat=2)]
        self.angle_list = ['-'.join(angle) for angle in itertools.product(element_list, repeat=3)]

        self.element_to_index = {elem: idx for idx, elem in enumerate(self.element_list)}
        self.pair_to_index = {pair: idx for idx, pair in enumerate(self.pair_list)}
        self.angle_to_index = {angle: idx for idx, angle in enumerate(self.angle_list)}

        self.parameter_dict = parameter_dict

        self.element_keys = self._get_keys(['element'])
        self.pair_keys = self._get_keys(['pair'])
        self.angle_keys = self._get_keys(['angle'])

        self.torch_parameters = self._convert_parameters_to_torch(self.element_list, self.element_keys, 'element')
        self.torch_pair_parameters = self._convert_parameters_to_torch(self.pair_list, self.pair_keys, 'pair')
        self.torch_angle_parameters = self._convert_parameters_to_torch(self.angle_list, self.angle_keys, 'angle')

    def _get_keys(self, categories: List[str]) -> Set[str]:

        keys = set()
        for category in categories:
            if category == 'element':
                for key in self.element_list:
                    if key in self.parameter_dict:
                        keys.update(self.parameter_dict[key].keys())
            elif category == 'pair':
                for key in self.pair_list:
                    if key in self.parameter_dict:
                        keys.update(self.parameter_dict[key].keys())
            elif category == 'angle':
                for key in self.angle_list:
                    if key in self.parameter_dict:
                        keys.update(self.parameter_dict[key].keys())
            else:
                logger.warning(f"Unknown category: {category}")
        logger.debug(f"Collected keys ({categories}): {keys}")
        return keys

    def _convert_parameters_to_torch(self, items: List[str], parameter_keys: Set[str], key_type: str) -> Dict[
        str, torch.Tensor]:
        
        extracted_parameters = {}
        for param_name in parameter_keys:
            if key_type == 'element':
                param_values = [self.parameter_dict.get(elem, {}).get(param_name, 0.0) for elem in items]
            elif key_type == 'pair':
                param_values = [
                    self.parameter_dict.get(pair, {}).get(param_name,
                                                          self.parameter_dict.get('-'.join(pair.split('-')[::-1]), {})
                                                          .get(param_name, 0.0))
                    for pair in items
                ]
            elif key_type == 'angle':
                param_values = [self.parameter_dict.get(angle, {}).get(param_name, 0.0) for angle in items]
            else:
                raise ValueError(f"Unsupported key_type: {key_type}")

            extracted_parameters[param_name] = torch.tensor(param_values, dtype=torch.float32).to(self.device)
            logger.debug(f"Converted parameter '{param_name}' ({key_type}): {extracted_parameters[param_name].shape}")
        return extracted_parameters

    def get_parameters_for_atoms(self, atom_types: List[str]) -> Dict[str, torch.Tensor]:

        try:
            atom_indices = torch.tensor([self.element_to_index[atom] for atom in atom_types], dtype=torch.long).to(
                self.device)
        except KeyError as e:
            logger.error(f"Unknown element type: {e}")
            raise ValueError(f"Unrecognized element type: {e}") from e

        parameter_tensors = {
            param_name: param_values[atom_indices]
            for param_name, param_values in self.torch_parameters.items()
        }
        logger.info("Get element parameters successfully")
        return parameter_tensors

    def get_parameters_for_pairs(self, atom_types: List[str]) -> Dict[str, torch.Tensor]:
        
        try:
            atom_pairs = [
                [self.pair_to_index[f"{atom_i}-{atom_j}"] for atom_j in atom_types]
                for atom_i in atom_types
            ]
        except KeyError as e:
            logger.error(f"Undefined pair parameters: {e}")
            raise ValueError(f"Missing parameters for atom pair: {e}") from e

        atom_pairs_tensor = torch.tensor(atom_pairs, dtype=torch.long).to(self.device)
        parameter_tensors = {
            param_name: param_values[atom_pairs_tensor]
            for param_name, param_values in self.torch_pair_parameters.items()
        }
        logger.info("Get pair parameters successfully")
        return parameter_tensors

    def get_parameters_for_angles(self, atom_types: List[str]) -> Dict[str, torch.Tensor]:
        
        atom_angles = [f"{atom_i}-{atom_j}-{atom_k}" for atom_i in atom_types for atom_j in atom_types for atom_k in
                       atom_types]
        angle_indices = [self.angle_to_index.get(angle, -1) for angle in atom_angles]

        if -1 in angle_indices:
            invalid_angles = [angle for angle, idx in zip(atom_angles, angle_indices) if idx == -1]
            logger.error(f"Undefined angle parameters: {invalid_angles}")
            raise ValueError("Missing parameters for angle")

        angle_indices_tensor = torch.tensor(angle_indices, dtype=torch.long).to(self.device)
        parameter_tensors = {
            param_name: param_values[angle_indices_tensor]
            for param_name, param_values in self.torch_angle_parameters.items()
        }
        logger.info("Get angle parameters successfully")
        return parameter_tensors
