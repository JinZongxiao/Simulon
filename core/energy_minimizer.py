import torch


def minimize_energy_bfgs_scipy(model, max_iter=20, lr=0.5, history_size=10, line_search_fn="strong_wolfe"):
    """
    Minimize the potential energy of the given model using PyTorch's LBFGS optimizer.

    Args:
        model: A molecular model with:
            - molecular.coordinates: torch.Tensor of shape (N,3), requires_grad set to True.
            - sum_bone(): returns dict with 'energy' scalar tensor.
        max_iter (int): Maximum LBFGS iterations.
        lr (float): Learning rate.
        history_size (int): Number of corrections in LBFGS.
        line_search_fn (str): Line search method.

    Returns:
        final_energy (float): The minimized energy.
    """
    coords = model.molecular.coordinates
    coords.requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [coords], lr=lr, max_iter=max_iter,
        history_size=history_size, line_search_fn=line_search_fn
    )

    def closure():
        optimizer.zero_grad()
        # Update neighbor list based on current coords
        try:
            model.molecular.update_coordinates(coords)
        except Exception:
            pass
        out = model.sum_bone()
        energy = out['energy']
        energy.backward()
        return energy

    print("Starting energy minimization (PyTorch LBFGS)...")
    optimizer.step(closure)
    with torch.no_grad():
        # Final neighbor update and energy report
        try:
            model.molecular.update_coordinates(coords)
        except Exception:
            pass
        final_energy = model.sum_bone()['energy'].item()
    print(f"Energy minimization finished, final energy: {final_energy:.6f}")
    return final_energy


def minimize_energy_steepest_descent(model, max_steps=10000, step_size=0.002, force_threshold=1e-2,
                                      print_interval=10, max_backtracks=12, min_step=1e-6):

    print("--- Starting Energy Minimization (Robust Steepest Descent) ---")

    coords = model.molecular.coordinates
    box_len = getattr(model.molecular, 'box_length', None)

    with torch.no_grad():
        try:
            model.molecular.update_coordinates(coords)
        except AttributeError:
            pass
        out = model.sum_bone()
        curr_energy = out['energy'].item()

    for step in range(max_steps):
        with torch.no_grad():
            out = model.sum_bone()
            energy = out['energy']
            forces = out['forces']

            max_force = torch.norm(forces, dim=1).max().item()
            if (step + 1) % print_interval == 0 or step == 0:
                print(f"Step {step+1}/{max_steps} -> Energy: {energy.item():.4f} eV, Max Force: {max_force:.6f} eV/Ang")

            if max_force < force_threshold:
                print(f"\nConvergence reached at step {step+1}.")
                print(f"Final Max Force ({max_force:.6f}) is below threshold ({force_threshold:.6f}).")
                curr_energy = energy.item()
                break

            force_norm = torch.norm(forces, dim=1, keepdim=True).clamp_min(1e-12)
            f_dir = forces / force_norm

            coords0 = coords.clone()
            base_energy = energy.item()

            step_try = float(step_size)
            accepted = False

            for _ in range(max_backtracks):
                for sign in (1.0, -1.0):
                    trial = coords0 + sign * step_try * f_dir

                    if box_len is not None:
                        trial = trial - torch.floor(trial / box_len) * box_len

                    try:
                        model.molecular.update_coordinates(trial)
                    except AttributeError:
                        pass
                    e_trial = model.sum_bone()['energy'].item()

                    if e_trial < base_energy:
                        coords.copy_(trial)
                        curr_energy = e_trial
                        accepted = True
                        break
                if accepted:
                    break
                step_try *= 0.5
                if step_try < min_step:
                    break

            if not accepted:
                try:
                    model.molecular.update_coordinates(coords0)
                except AttributeError:
                    pass
                print("Line search failed to find a descending step; early stop.")
                curr_energy = base_energy
                break

    with torch.no_grad():
        out_final = model.sum_bone()
        final_energy = out_final['energy'].item()
        final_max_force = torch.norm(out_final['forces'], dim=1).max().item()

    print(f"--- Steepest Descent Finished ---")
    print(f"Final Energy: {final_energy:.4f} eV, Final Max Force: {final_max_force:.6f} eV/Ang")
    return final_energy
