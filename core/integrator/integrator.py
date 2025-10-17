import torch.nn as nn
import torch
from core.md_model import IntegratorInterface

class VerletIntegrator(IntegratorInterface, nn.Module):
    def __init__(self,
                 molecular,
                 dt,
                 force_field=None, 
                 ensemble=None, temperature=None, gamma=None):
        super().__init__()
        self.molecular = molecular
        self.box_length = molecular.box_length
        self.force_field = force_field 
        
        self.BOLTZMAN = 8.617333262e-5  # ev/K
        self.N_a = 6.02214076e23  # 1/mol
        self.J_per_ev = 1.60218e-19  # J/eV
        
        self.dt = dt
        self.atom_mass = self.molecular.atom_mass.unsqueeze(-1).expand_as(self.molecular.atom_velocities)
        self.atom_mass = self.atom_mass * (10.0 / (self.N_a * self.J_per_ev))
        
        
        if ensemble == 'NVT' and temperature is not None and gamma is not None:
            self.init_temperature = torch.tensor(temperature[0], device=molecular.device)
            self.temperature = torch.tensor(temperature[1], device=molecular.device)
            self.gamma = torch.tensor(gamma, device=molecular.device)
            self.is_langevin_thermostat = True
            self.molecular.set_maxwell_boltzmann_velocity(self.init_temperature)
            #VELOCITY_CONVERSION = 9.79589e-1
            # self.random_force_factor = torch.sqrt(
            #     2 * self.gamma * self.BOLTZMAN * self.temperature / self.atom_mass *
            #     (1 - torch.exp(-2 * self.gamma * self.dt)) / (2 * self.gamma)
            # ) * VELOCITY_CONVERSION
            exp_term = torch.exp(-2.0 * self.gamma * self.dt)
            self.random_force_factor = torch.sqrt((self.BOLTZMAN * self.temperature) / self.atom_mass * (1.0 - exp_term))
        else:
            self.is_langevin_thermostat = False
        self.new_coords = torch.empty_like(molecular.coordinates,device=molecular.device)
        self.vel_half = torch.zeros_like(molecular.coordinates,device=molecular.device)

    def apply_pbc(self, coordinates):
        return coordinates - torch.floor(coordinates / self.box_length) * self.box_length

    def first_half(self, forces_old: torch.Tensor):
        vel = self.molecular.atom_velocities
        accel = forces_old / self.atom_mass
        self.vel_half = vel + 0.5 * accel * self.dt
        self.new_coords = self.molecular.coordinates + self.vel_half * self.dt
        self.molecular.update_coordinates(self.new_coords)

    def second_half(self, forces_new: torch.Tensor):
        if self.is_langevin_thermostat:
            csi = torch.randn_like(self.random_force_factor) * self.random_force_factor
            damping_factor = torch.exp(-self.gamma * self.dt)
            self.vel_half = self.vel_half * damping_factor + csi
        accel_new = forces_new / self.atom_mass
        vel = self.vel_half + 0.5 * accel_new * self.dt
        self.molecular.update_velocities(vel)
        kin_energy = (0.5 * self.atom_mass * vel.pow(2)).sum()
        T = (2.0 / 3.0) * kin_energy / (self.molecular.atom_count * self.BOLTZMAN)
        return {'update_coordinates': self.new_coords,
                'kinetic_energy': kin_energy,
                'temperature': T}

    def forward(self, force):
        vel = self.molecular.atom_velocities
        accel = force / self.atom_mass
        self.new_coords = self.molecular.coordinates + vel * self.dt + 0.5 * accel * (self.dt ** 2)
        self.vel_half = vel + 0.5 * accel * self.dt
        self.molecular.update_coordinates(self.new_coords)
        if self.force_field is not None:
            force = self.force_field()['forces']
        if self.is_langevin_thermostat:
            csi = torch.randn_like(self.random_force_factor) * self.random_force_factor
            damping_factor = torch.exp(-self.gamma * self.dt)
            self.vel_half = self.vel_half * damping_factor + csi
        accel = force / self.atom_mass
        vel = self.vel_half + 0.5 * accel * self.dt
        self.molecular.update_velocities(vel)
        kin_energy = (0.5 * self.atom_mass * vel.pow(2)).sum()
        T = (2.0 / 3.0) * kin_energy / (self.molecular.atom_count * self.BOLTZMAN)
        return {'update_coordinates': self.new_coords,
                'kinetic_energy': kin_energy,
                'temperature': T}
