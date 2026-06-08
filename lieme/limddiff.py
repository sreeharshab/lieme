import os
import re
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from ase.io import read, write, Trajectory
from mp_api.client import MPRester
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class LiMDDiffusion:
    def __init__(self,
                 api_key: Optional[str]=None,
                 root_dir: Optional[str]=None,
                 poscar_dir: str="POSCARs",
                 outs_dir: str="outs",
                 model_path: str="/expanse/lustre/projects/cla175/sbhimineni/Non_Li_systems/mace-mpa-0-medium.model-lammps.pt",
                 lammps_command: str="/home/sbhimineni/lammps_with_mace_gpu/lammps/build/lmp -in in.diffusion",
                 n_runs: int=2,
                 temperatures: List[int]=[2000],
                 timestep: float=0.001,
                 n_atoms: int=40,
                 partition: str="gpu-shared",
                 nodes: int=1,
                 ntasks_per_node: int=24,
                 gpus: int=1,
                 mem: str="96G",
                 account: str="cla346",
                 walltime: str="48:00:00",
                 exclude: Optional[str]="exp-9-60",
                 conda_env: str="lieme"
                 ):
        """High-throughput workflow for Li-containing Materials Project compounds."""
        self.api_key = self._resolve_api_key(api_key)
        self.root_dir = root_dir or os.getcwd()
        self.poscar_dir = os.path.join(self.root_dir, poscar_dir)
        self.outs_dir = os.path.join(self.root_dir, outs_dir)
        self.model_path = model_path
        self.lammps_command = lammps_command
        self.n_runs = n_runs
        self.temperatures = temperatures
        self.timestep = timestep
        self.n_atoms = n_atoms
        self.partition = partition
        self.nodes = nodes
        self.ntasks_per_node = ntasks_per_node
        self.gpus = gpus
        self.mem = mem
        self.account = account
        self.walltime = walltime
        self.exclude = exclude
        self.conda_env = conda_env

    def _resolve_api_key(self, api_key: Optional[str]) -> str:
        if api_key:
            return api_key
        env_key = os.getenv("MP_API_KEY") or os.getenv("MAPI_KEY")
        if not env_key:
            raise ValueError("Materials Project API key not provided. Set MP_API_KEY or pass api_key.")
        return env_key

    def _sanitize_name(self, name: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]", "_", name)

    def _select_materials_by_indices(self,
                                     materials: List[Dict[str, str]],
                                     indices: Optional[List[int]]=None
                                     ) -> List[Dict[str, str]]:
        if not indices:
            return materials
        selected = []
        seen = set()
        n_materials = len(materials)
        invalid = []
        for raw_idx in indices:
            idx = raw_idx
            if idx < 0:
                idx = n_materials + idx
            if idx < 0 or idx >= n_materials:
                invalid.append(raw_idx)
                continue
            if idx in seen:
                continue
            seen.add(idx)
            selected.append(materials[idx])
        if invalid:
            logging.warning(f"Ignoring out-of-range indices: {sorted(set(invalid))}")
        return selected

    def fetch_li_materials(self, indices: Optional[List[int]]=None) -> List[Dict[str, str]]:
        """Fetch Li-containing compounds from Materials Project."""
        materials = []
        used_names = set()
        with MPRester(self.api_key) as mpr:
            docs = list(mpr.materials.summary.search(
                elements=["Li"],
                theoretical=False,
                fields=["material_id", "formula_pretty", "theoretical"],
                num_chunks=None,
                chunk_size=1000
            ))
            docs = [doc for doc in docs if not getattr(doc, "theoretical", False)]
            docs = self._select_materials_by_indices(docs, indices)
            for count, doc in enumerate(docs):
                try:
                    structure = mpr.get_structure_by_material_id(doc.material_id)
                except Exception as e:
                    logging.warning(f"Skipping {doc.material_id}: failed fetching structure ({e})")
                    continue
                if structure is None:
                    logging.warning(f"Skipping {doc.material_id}: no structure returned")
                    continue
                if isinstance(structure, dict):
                    try:
                        structure = Structure.from_dict(structure)
                    except Exception as e:
                        logging.warning(f"Skipping {doc.material_id}: invalid structure dict ({e})")
                        continue
                atoms = AseAtomsAdaptor.get_atoms(structure)
                base_name = self._sanitize_name(doc.formula_pretty)
                material_name = base_name
                if material_name in used_names:
                    material_name = f"{base_name}_{doc.material_id}"
                used_names.add(material_name)
                materials.append({
                    "index": indices[count] if indices else count,
                    "material_id": doc.material_id,
                    "formula_pretty": doc.formula_pretty,
                    "material_name": material_name,
                    "atoms": atoms
                })
        return materials

    def submit_jobs(
            self, 
            indices: Optional[List[int]]=None,
            batch_size: Optional[int]=None,
            batch_starts: Optional[List[int]]=None
            ):
        """Create POSCARs, job scripts, and submit diffusion jobs for Li-containing materials."""
        os.makedirs(self.poscar_dir, exist_ok=True)
        os.makedirs(self.outs_dir, exist_ok=True)
        if indices and not batch_size:
            materials = self.fetch_li_materials(indices=indices)
        elif batch_size and batch_starts and not indices:
            batch_indices = []
            for start in batch_starts:
                for i in range(batch_size):
                    batch_indices.append(start + i)
            materials = self.fetch_li_materials(indices=batch_indices)
        if not materials:
            logging.warning("No Li-containing materials found in Materials Project.")
            return
        if not indices and not batch_starts and batch_size:
            batch_starts = list(range(0, len(materials), batch_size))
        if batch_size:
            batch_materials_info = []
        for material in materials:
            material_name = material["material_name"]
            material_index = material["index"]
            atoms = material.get("atoms")
            if atoms is None:
                logging.warning(f"Skipping {material_name}: no atoms in materials list")
                continue
            poscar_file = os.path.join(self.poscar_dir, f"{material_name}.poscar")
            write(poscar_file, atoms)
            if batch_size:
                batch_materials_info.append((material_index, material_name, poscar_file))
            if indices and not batch_size:
                logging.info(f"Submitting job for material: {material_name}")
                run_script_content = f"""from ase.io import read
from mace.calculators import mace_mp
from lieme.workflowgen import WorkFlowGenerator

if __name__ == "__main__":
    atoms = read("{poscar_file}")
    mlip_calc = None
    workflow = WorkFlowGenerator(material="{material_name}", atoms=atoms, mlip_calc=mlip_calc)
    workflow.run_diffusion_calc(atoms=atoms, n_runs={self.n_runs}, temperatures={self.temperatures}, timestep={self.timestep}, n_atoms={self.n_atoms}, model="{self.model_path}")
"""
                run_script_name = f"run_{material_name}.py"
                with open(run_script_name, "w") as f:
                    f.write(run_script_content)

                exclude_line = f"#SBATCH --exclude={self.exclude}\n" if self.exclude else ""
                job_script_content = f"""#!/bin/bash
#SBATCH --output="outs/{material_name}.out"
#SBATCH --export=ALL
#SBATCH --partition={self.partition}
#SBATCH --nodes={self.nodes}
#SBATCH --ntasks-per-node={self.ntasks_per_node}
#SBATCH --gpus={self.gpus}
#SBATCH --mem={self.mem}
#SBATCH --account={self.account}
#SBATCH -t {self.walltime}
{exclude_line}
source ~/.bashrc
conda activate {self.conda_env}

module load gpu
module load gcc/10.2.0/i62tgso
module load cuda12.2/toolkit
module load openmpi/4.1.3/gzzscfu
module load intel-mkl/2020.4.304/n3sprct-omp

export OMP_NUM_THREADS=${{SLURM_NTASKS}}
export LAMMPS_COMMAND="{self.lammps_command}"

python {run_script_name}
"""
                job_script_name = f"job_{material_name}.sh"
                with open(job_script_name, "w") as f:
                    f.write(job_script_content)
                result = subprocess.run(["sbatch", job_script_name], capture_output=True, text=True)
                if result.stdout.strip():
                    logging.info(result.stdout.strip())
                if result.returncode != 0:
                    logging.error(result.stderr.strip())
        if batch_size:
            for start in batch_starts:
                batch = [(info[1], info[2]) for info in batch_materials_info if info[0] >= start and info[0] < start + batch_size]
                if not batch:
                    continue
                run_script_content = f"""from ase.io import read
from mace.calculators import mace_mp
from lieme.workflowgen import WorkFlowGenerator

if __name__ == "__main__":
    batch = {batch}
    for material_name, poscar_file in batch:
        atoms = read(poscar_file)
        workflow = WorkFlowGenerator(material=material_name, atoms=atoms)
        workflow.run_diffusion_calc(atoms=atoms, n_runs={self.n_runs}, temperatures={self.temperatures}, timestep={self.timestep}, n_atoms={self.n_atoms}, model="{self.model_path}")
"""
                run_script_name = f"run_batch_{start}.py"
                with open(run_script_name, "w") as f:
                    f.write(run_script_content)
                exclude_line = f"#SBATCH --exclude={self.exclude}\n" if self.exclude else ""
                job_script_content = f"""#!/bin/bash
#SBATCH --output="outs/batch_{start}.out"
#SBATCH --export=ALL
#SBATCH --partition={self.partition}
#SBATCH --nodes={self.nodes}
#SBATCH --ntasks-per-node={self.ntasks_per_node}
#SBATCH --gpus={self.gpus}
#SBATCH --mem={self.mem}
#SBATCH --account={self.account}
#SBATCH -t {self.walltime}
{exclude_line}
source ~/.bashrc
conda activate {self.conda_env}

module load gpu
module load gcc/10.2.0/i62tgso
module load cuda12.2/toolkit
module load openmpi/4.1.3/gzzscfu
module load intel-mkl/2020.4.304/n3sprct-omp

export OMP_NUM_THREADS=${{SLURM_NTASKS}}
export LAMMPS_COMMAND="{self.lammps_command}"

python {run_script_name}
"""
                job_script_name = f"job_batch_{start}.sh"
                with open(job_script_name, "w") as f:
                    f.write(job_script_content)
                result = subprocess.run(["sbatch", job_script_name], capture_output=True, text=True)
                if result.stdout.strip():
                    logging.info(result.stdout.strip())
                if result.returncode != 0:
                    logging.error(result.stderr.strip())

    def collect_trajectories(self,
                             output_path: str="all.traj",
                             patterns: Optional[List[str]]=None):
        """Collect all Li-containing trajectories into a single ASE trajectory file."""
        if patterns is None:
            patterns = ["gcbh.traj", "equil.lammpstrj", "diffuse.lammpstrj"]
        root = Path(self.root_dir).resolve()
        traj = Trajectory(os.path.join(self.root_dir, output_path), "w")
        matches = []
        for name in patterns:
            matches.extend(root.rglob(name))
        for path in sorted(set(matches)):
            try:
                traj_read = Trajectory(str(path), "r")
            except Exception:
                traj_read = read(str(path), format="lammps-dump-text", index=":")
            for atoms in traj_read:
                if "Li" in atoms.get_chemical_symbols():
                    traj.write(atoms)
        traj.close()
