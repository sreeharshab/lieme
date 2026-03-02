import os
import re
import shutil
import logging
import subprocess
from typing import List, Dict, Optional
import numpy as np
from ase import Atoms
from ase.io import read, write, Trajectory
from ase.calculators.calculator import Calculator
from ase.calculators.vasp import Vasp
from lieme.io import repeat_to_n_atoms
from lieme.featurize import GetFeatures, relax

"""
Run order: mlip_calcs -> energy_calc -> electronic_calc -> bader_calc -> intercalation_calc -> diffusion_calc
May fail if order is changed!
"""

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class WorkFlowGenerator:
    def __init__(self,
                 material: str,
                 atoms: Atoms,
                 mlip_calc: Optional[Calculator]=None,
                 fmax: float=0.05,
                 u: Optional[Dict]=None,
                 dk: float=20,
                 addnl_dft_settings: Optional[Dict]=None,
                 deintercalate: bool=False,
                 remove_li_m_ratio: Optional[float]=None,
                 remove_sampling_size: int=50,
                 li_m_ratios: List[float]=[0.25],
                 li_m_ratio_tol: float=0.1,
                 custom_n_m: Optional[Dict]=None,
                 sampling_size: int=50,
                 seed: int=10,
                 li_atom_cutoff: float=1.7,
                 li_li_cutoff: float=1.0
                 ):
        """Generates workflow for MLIP and DFT calculations which are used by GetFeatures to obtain material features.

        Args:
            material (str): Name of the material. This will be the name of the calculation directory.
            atoms (Atoms): ASE Atoms object of the material.
            mlip_calc (Calculator, optional): MLIP ASE Calculator object to be used for initial intercalation sampling.
                Make sure to pass a calculator if `deintercalate` is True or `run_mlip_calcs` method is used. 
                Defaults to None.
            fmax (float, optional): Force convergence criterion for geometry optimizations. Defaults to 0.05 eV/Å.
            u (Optional[Dict], optional): Hubbard U values for the atoms in the material. Defaults to None.
            dk (float, optional): k-point density in reciprocal space for DFT calculations. Defaults to 20 1/Å.
            addnl_dft_settings (Optional[Dict], optional): Additional settings to add to or replace default 
                VASP calculation settings. Defaults to None.
            deintercalate (bool, optional): Whether to remove Li from the parent structure if it contains Li.
                For example, LiCoO2 to LixCoO2. Defaults to False.
            remove_li_m_ratio (Optional[float], optional): If `deintercalate` is True, the ratio of Li to metal 
                atoms to be removed from the parent structure. Defaults to None.
            remove_sampling_size (int, optional): If `deintercalate` is True, the number of random deintercalated 
                structures to sample. Defaults to 50.
            Check GetFeatures.get_intercalation_data() for documentation of the remaining parameters.
        """
        self.material = material
        self.root_dir = os.getcwd()
        self.material_dir = os.path.join(self.root_dir, self.material)
        self.seed = seed
        if len(atoms) < 20:
            logging.warning(f"The provided Atoms object for {material} has less than 10 atoms. "
                            "Repeating it to avoid errors during Li intercalation/deintercalation" \
                            " and feature extraction.")
            atoms = repeat_to_n_atoms(atoms, n_atoms=20)
        self.atoms = atoms
        self.mlip_calc = mlip_calc
        self.fmax = fmax
        if deintercalate:
            self.remove_li_m_ratio = remove_li_m_ratio
            self.remove_sampling_size = remove_sampling_size
            self.get_best_deintercalated_structure()
        self.u = u
        self.dk = dk
        self.addnl_dft_settings = addnl_dft_settings
        self.li_m_ratios = li_m_ratios
        self.li_m_ratio_tol = li_m_ratio_tol
        self.custom_n_m = custom_n_m
        self.sampling_size = sampling_size
        self.li_atom_cutoff = li_atom_cutoff
        self.li_li_cutoff = li_li_cutoff    
    
    def create_dirs(self) -> Dict[str, str]:
        material_dir = self.material_dir
        dirs = {
            "material": material_dir,
            "energy": os.path.join(material_dir, "Energy_calculation"),
            "electronic": os.path.join(material_dir, "Electronic_calculation"),
            "bader": os.path.join(material_dir, "Bader_calculation"),
            "intercalation": os.path.join(material_dir, "Intercalation"),
            "diffusion": os.path.join(material_dir, "Diffusion_calculation")
        }
        for dir in dirs.values():
            os.makedirs(dir, exist_ok=True)
        self.dirs = dirs
        return dirs
    
    def get_best_deintercalated_structure(self) -> Atoms:
        """This method is used to remove Li from materials which contain Li in their parent structure,
            for example, LiCoO2. Typically, this deintercalated structure, for example, LixCoO2, will 
            act as a parent structure for all calculations in these kind of materials.
        """
        deintercalation_dir = os.path.join(self.material_dir, "Deintercalation")
        os.makedirs(deintercalation_dir, exist_ok=True)
        os.chdir(deintercalation_dir)
        traj = Trajectory(f"sampling_{self.remove_li_m_ratio}.traj", "w")
        atoms = self.atoms
        seed = self.seed
        np.random.seed(seed)
        li_indices = [i for i, atom in enumerate(atoms) if atom.symbol == "Li"]
        n_li = len(li_indices)
        if n_li == 0:
            logging.warning("No Li atoms found in the structure. Skipping deintercalation step.")
            os.chdir(self.root_dir)
            return
        n_remove = int(round(n_li*self.remove_li_m_ratio))
        atoms_without_li_dict = {}
        for i in range(self.remove_sampling_size):
            atoms_without_li = atoms.copy()
            idxs = np.random.choice(li_indices, size=n_remove, replace=False)
            del atoms_without_li[idxs]
            atoms_without_li = relax(atoms=atoms_without_li, calc=self.mlip_calc, fmax=self.fmax)
            atoms_without_li_dict[i] = atoms_without_li
            traj.write(atoms_without_li)
        traj.close()
        sorted_i = sorted(atoms_without_li_dict.keys(), key=lambda x: atoms_without_li_dict[x].get_potential_energy())
        self.atoms = atoms_without_li_dict[sorted_i[0]]
        os.chdir(self.root_dir)
        return

    def run_mlip_calcs(self) -> Atoms:
        features = GetFeatures(
            material=self.material,
            atoms=self.atoms,
            calc=self.mlip_calc,
            fmax=self.fmax
        )
        relaxed_atoms = features.relaxed_atoms
        self.relaxed_atoms = relaxed_atoms
        self.metals = features.metals
        features.get_intercalation_data(
            li_m_ratios=self.li_m_ratios,
            li_m_ratio_tol=self.li_m_ratio_tol,
            custom_n_m=self.custom_n_m,
            sampling_size=self.sampling_size,
            seed=self.seed,
            li_atom_cutoff=self.li_atom_cutoff,
            li_li_cutoff=self.li_li_cutoff
        )
        features.return_to_root()
        return relaxed_atoms

    def get_ldau_luj(self) -> Dict[int, List[float]]:
        u = self.u
        ldau_luj = {}
        symbols = set(self.atoms.get_chemical_symbols())
        us = {  # Materials Project recommended U values
                # https://docs.materialsproject.org/methodology/materials-methodology/calculation-details/gga+u-calculations/hubbard-u-values
            'Co': 3.32,
            'Cr': 3.7,
            'Fe': 5.3,
            'Mn': 3.9,
            'Mo': 4.38,
            'Ni': 6.2,
            'V': 3.25,
            'W': 6.2
        }
        if not u:
            u = {}
            for symbol in symbols:
                if symbol in us.keys() and "O" in symbols:
                    u[symbol] = us[symbol]
        for symbol in symbols:
            if symbol in u.keys():
                ldau_luj[symbol] = {"L": 2, "U": u[symbol], "J": 0.0}
            else:
                ldau_luj[symbol] = {"L": -1, "U": 0.0, "J": 0.0}
        return ldau_luj

    def generate_kpts(self, atoms: Atoms, dk: float=20) -> List[int]:
        reciprocal_cell = atoms.cell.reciprocal()
        b_lengths = [np.linalg.norm(v) for v in reciprocal_cell]
        kpts = [max(1, int(np.ceil(length*dk))) for length in b_lengths]
        return kpts
 
    def get_valence_electrons(self, 
                              atoms: Atoms, 
                              ) -> Dict[str, int]:
        # nelects are obtained from Materials Project recommended POTCARs present in either potpaw, 
        # potpaw_PBE, or potpaw_GGA directories based on the gga setting.
        # Make sure not to replace gga setting in the base_calc with addnl_dft_settings!
        pots = {   # Materials Project recommended POTCARs obtained from ASE 
                   # (https://gitlab.com/ase/ase/-/blob/master/ase/calculators/vasp/setups.py?ref_type=heads)
            'Li': '_sv',
            'Na': '_pv',
            'K': '_sv',
            'Cs': '_sv',
            'Rb': '_sv',
            'Be': '_sv',
            'Mg': '_pv',
            'Ca': '_sv',
            'Sr': '_sv',
            'Ba': '_sv',
            'Sc': '_sv',
            'Y': '_sv',
            'Ti': '_pv',
            'Zr': '_sv',
            'Hf': '_pv',
            'V': '_sv',
            'Nb': '_pv',
            'Ta': '_pv',
            'Cr': '_pv',
            'Mo': '_pv',
            'W': '_sv',
            'Mn': '_pv',
            'Tc': '_pv',
            'Re': '_pv',
            'Fe': '_pv',
            'Co': '',
            'Ni': '_pv',
            'Cu': '_pv',
            'Zn': '',
            'Ru': '_pv',
            'Rh': '_pv',
            'Pd': '',
            'Ag': '',
            'Cd': '',
            'Hg': '',
            'Ir': '',
            'Pt': '',
            'Os': '_pv',
            'Ga': '_d',
            'Ge': '_d',
            'Al': '',
            'As': '',
            'Se': '',
            'Br': '',
            'In': '_d',
            'Sn': '_d',
            'Tl': '_d',
            'Pb': '_d',
            'Bi': '_d',
            'Po': '',
            'At': '_d',
            'La': '',
            'Ce': '',
            'Pr': '_3',
            'Nd': '_3',
            'Pm': '_3',
            'Sm': '_3',
            'Eu': '',
            'Gd': '',
            'Tb': '_3',
            'Dy': '_3',
            'Ho': '_3',
            'Er': '_3',
            'Tm': '_3',
            'Yb': '_3',
            'Lu': '_3'
        }
        valence_electrons = {}
        base_calc = self.get_base_calc()
        potcar_folder = os.getenv("VASP_PP_PATH")
        def get_potcar_subfolder(settings):
            if settings["gga"]==None:
                potcar_subfolder = "potpaw"
            elif settings["gga"]=="PE":
                potcar_subfolder = "potpaw_PBE"
            elif settings["gga"]=="91":
                potcar_subfolder = "potpaw_GGA"
            return potcar_subfolder
        if potcar_folder is not None:
            formula = str(atoms.symbols)
            elements = re.findall(r'([A-Z][a-z]?)\d*', formula)
            if "setups" in base_calc.parameters:
                setups = base_calc.parameters["setups"]
                keys = list(setups.keys())
            else:
                keys = None
            potcar_subfolder = get_potcar_subfolder(base_calc.parameters)
            cwd = os.getcwd()
            os.chdir(os.path.join(potcar_folder, potcar_subfolder))
            for element in elements:
                if keys is not None and element in keys:
                    os.chdir(f"{element}"+f"{setups[element]}")
                elif element in pots.keys():
                    os.chdir(f"{element}"+pots[element])
                else:
                    os.chdir(element)
                temp_f = open("POTCAR", "r")
                lines = temp_f.readlines()
                search_str = lines[0].split()[0]
                for i,line in enumerate(lines):
                    if search_str in line and not any(excluded in line 
                                                      for excluded in 
                                                      ["TITEL", "LPAW", "radial sets"]):
                        match = re.search(fr"{search_str}\s+([A-Z][a-z]?)", line)
                        next_line = lines[i+1]
                        nelect = int(float(next_line.split()[0]))
                    if match:
                        element = match.group(1)
                    valence_electrons[element] = nelect
                temp_f.close()
                os.chdir("../")
            os.chdir(cwd)
        else:
            raise EnvironmentError("VASP_PP_PATH environment variable not set. "
            "Cannot determine valence electrons without access to POTCAR files.")
        return valence_electrons

    def get_base_calc(self) -> Vasp:
        base_calc = Vasp(
            gga="PE",
            setups={"base": "materialsproject"},
            lreal="Auto",
            lplane=True,
            lwave=False,
            lcharg=False,
            gamma=True,
            ncore=8,
            algo="Normal",
            encut=520,
            ediff=1e-5,
            nelm=200,
            ibrion=2,
            ediffg=-self.fmax,
            nsw=200,
            isif=3,
            ismear=0,
            sigma=0.05,
            ispin=2,
            lvdw=True,
            ivdw=12,
            ldau=True,
            ldautype=2,
            ldau_luj = self.get_ldau_luj(),
            lmaxmix=4,
        )
        if self.addnl_dft_settings is not None:
            # Additional settings will override defaults if there are conflicts!
            for key, value in self.addnl_dft_settings.items():
                base_calc.set(**{key: value})
        return base_calc
    
    def run_energy_calc(self, atoms: Optional[Atoms]=None, dir: Optional[str]=None):
        if not dir:
            dir = self.dirs["energy"]
        if not atoms:
            try:
                atoms = self.relaxed_atoms.copy()
            except AttributeError:
                try:
                    atoms = read(os.path.join(dir, "opt.traj@-1"))
                except FileNotFoundError:
                    atoms = self.atoms.copy()
        os.chdir(dir)
        if os.path.exists("opt2.outcar"):
            with open("opt2.outcar", "r") as f:
                f.seek(0, 2)
                f_size = f.tell()
                read_size = min(1000, f_size)
                f.seek(max(0, f_size - read_size))
                end_content = f.read()
                if "General timing and accounting informations for this job:" in end_content:
                    logging.info(f"OUTCAR already exists. Skipping DFT {dir.split("/")[-1]}...")
                    self.relaxed_atoms = read("OUTCAR", index=-1)
                    return
        calc = self.get_base_calc()
        levels = 3
        for i in range(levels):
            kpts = self.generate_kpts(atoms, dk=np.floor((self.dk/levels)*(i+1)))
            calc.set(kpts=kpts)
            atoms.calc = calc
            atoms.get_potential_energy()
            shutil.copy("CONTCAR", f"opt{i}.vasp")
            shutil.copy("OUTCAR", f"opt{i}.outcar")
            shutil.copy("vasp.out", f"opt{i}.out")
            shutil.copy("vasprun.xml", f"opt{i}.xml")
        self.relaxed_atoms = atoms.copy()
        os.chdir(self.root_dir)

    def run_electronic_calc(self, atoms: Optional[Atoms]=None, dir: Optional[str]=None, n_addnl_bands: int=20):
        if not atoms:
            atoms = self.relaxed_atoms.copy()
        if not dir:
            dir = self.dirs["electronic"]
        os.chdir(dir)
        complete = False
        if os.path.exists("spc.outcar"):
            with open("spc.outcar", "r") as f:
                f.seek(0, 2)
                f_size = f.tell()
                read_size = min(1000, f_size)
                f.seek(max(0, f_size - read_size))
                end_content = f.read()
                if "General timing and accounting informations for this job:" in end_content:
                    complete = True
        if not complete:
            calc = self.get_base_calc()
            kpts = self.generate_kpts(atoms, dk=self.dk+20) # Higher dk for better DOS
            calc.set(kpts=kpts, ibrion=-1, nsw=0, ismear=-5, lcharg=True, lwave=True)
            atoms.calc = calc
            atoms.get_potential_energy()
            os.rename("OUTCAR", "spc.outcar")
            os.rename("vasp.out", "spc.out")
        complete = False
        if os.path.exists("dos.outcar") and os.path.exists("dos.out"):
            with open("dos.outcar", "r") as f:
                f.seek(0, 2)
                f_size = f.tell()
                read_size = min(1000, f_size)
                f.seek(max(0, f_size - read_size))
                end_content = f.read()
                if "General timing and accounting informations for this job:" in end_content:
                    complete = True
                    logging.info(f"dos.outcar already exists. Skipping {dir.split("/")[-1]}...")
        if not complete:
            calc = self.get_base_calc()
            kpts = self.generate_kpts(atoms, dk=self.dk+20)
            valence_electrons = self.get_valence_electrons(atoms)
            nelect = 0
            for atom in atoms:
                nelect += valence_electrons[atom.symbol]
            nbands = nelect + n_addnl_bands
            calc.set(
                kpts=kpts,
                ibrion=-1,
                nsw=0,
                ismear=-5,
                icharg=11,
                lorbit=11,
                nedos=3000,
                emax=15,
                emin=-20,
                nbands=nbands
                )
            atoms.calc = calc
            atoms.get_potential_energy()
            os.rename("OUTCAR", "dos.outcar")
            os.rename("vasp.out", "dos.out")
        os.chdir(self.root_dir)

    def run_bader_calc(self, atoms: Optional[Atoms]=None, dir: Optional[str]=None):
        if not atoms:
            atoms = self.relaxed_atoms.copy()
        if not dir:
            dir = self.dirs["bader"]
        os.chdir(dir)
        if os.path.exists("AECCAR0") and os.path.exists("AECCAR2"):
            pass
        else:
            calc = self.get_base_calc()
            kpts = self.generate_kpts(atoms, dk=self.dk)
            calc.set(
                kpts=kpts,
                ibrion=-1,
                nsw=0,
                lorbit=12,
                lcharg=True,
                laechg=True,
            )
            atoms.calc = calc
            atoms.get_potential_energy()
        if os.path.exists("ACF.dat"):
            logging.info(f"ACF.dat already exists. Skipping {dir.split("/")[-1]}...")
            pass
        else:
            chgsum = os.getenv("VTST_SCRIPTS") + "/chgsum.pl"
            assert os.path.exists(chgsum), "chgsum.pl not found, export VTST_SCRIPTS "
            "environment variable pointing to VTST scripts directory."
            subprocess.run([chgsum, "AECCAR0", "AECCAR2"], capture_output=True)
            bader = os.getenv("VTST_BADER")
            assert os.path.exists(bader), "Bader executable not found, export VTST_BADER "
            "environment variable pointing to Bader executable."
            subprocess.run([bader, "CHGCAR", "-ref", "CHGCAR_sum"], capture_output=True)
        os.chdir(self.root_dir)

    def run_intercalation_calc(self, n_lowest: int=3):
        os.chdir(self.dirs["intercalation"])
        nlidirs = [entry.name for entry in os.scandir("./") if entry.is_dir()]
        nlidirs = sorted(nlidirs, key=lambda x: float(x.split("_")[0]))
        li_m_ratios = sorted(self.li_m_ratios)
        self.best_atoms_with_li = {}
        for idx, nlidir in enumerate(nlidirs):
            traj = read(f"sampling_{li_m_ratios[idx]}.traj", index=":")
            sub_dir = os.path.join(self.dirs["intercalation"], nlidir)
            os.chdir(sub_dir)
            mlip_energies = {}
            for i, atoms in enumerate(traj):
                mlip_energies[i] = atoms.get_potential_energy()
            sorted_i_mlip = sorted(mlip_energies.keys(), key=lambda x: mlip_energies[x])
            sub_samples = [str(i) for i in sorted_i_mlip[:n_lowest]]
            dft_atoms_with_li = {}
            for sample in sub_samples:
                sub_sub_dir = os.path.join(sub_dir, sample)
                geo_opt_dir = os.path.join(sub_sub_dir, "geo_opt")
                dos_dir = os.path.join(sub_sub_dir, "dos")
                bader_dir = os.path.join(sub_sub_dir, "bader")
                os.makedirs(dos_dir, exist_ok=True)
                os.makedirs(bader_dir, exist_ok=True)
                atoms_with_li = read(os.path.join(geo_opt_dir, "opt.traj@-1"))
                self.run_energy_calc(atoms=atoms_with_li, dir=geo_opt_dir) # This replaces self.relaxed_atoms with the intercalated structure!
                dft_atoms_with_li[sample] = self.relaxed_atoms
                self.run_electronic_calc(atoms=self.relaxed_atoms, dir=dos_dir)
                self.run_bader_calc(atoms=self.relaxed_atoms, dir=bader_dir)
            sorted_i_dft = sorted(dft_atoms_with_li.keys(), key=lambda x: dft_atoms_with_li[x].get_potential_energy())
            best_sample = sorted_i_dft[0]
            self.best_atoms_with_li[li_m_ratios[idx]] = dft_atoms_with_li[best_sample]
            rejected_samples = [str(i) for i in sorted_i_mlip[n_lowest:]]
            # Removing non-DFT relaxed samples to increase efficiency
            for sample in rejected_samples:
                if os.path.exists(os.path.join(sub_dir, sample)):
                    shutil.rmtree(os.path.join(sub_dir, sample))
        os.chdir(self.root_dir)
    
    def setup_md(self,
                 li_m_ratio: float,
                 n_atoms: int=200,
                 temperatures: list=[1000],
                 lammps_in_script: Optional[str]=None,
                 timestep: float=0.002,  # Time unit in ps
                 n_steps_equil: int=10000,
                 n_steps_diffuse: int=50000,
                 model: Optional[str]=None,
                 n_runs: int=3,
                 ):
        os.chdir(self.dirs["diffusion"])
        atoms_with_li = self.best_atoms_with_li[li_m_ratio]
        atoms_with_li = repeat_to_n_atoms(atoms_with_li, n_atoms=n_atoms)
        ratio_dir = os.path.join(self.dirs["diffusion"], f"{li_m_ratio}_Li_M")
        os.makedirs(ratio_dir, exist_ok=True)
        for temperature in temperatures:
            for run in range(n_runs):
                vrand = 4928459 + run
                md_dir = os.path.join(ratio_dir ,f"{temperature}/run_{run+1}")
                os.makedirs(md_dir, exist_ok=True)
                write(f"{md_dir}/data.structure", 
                                    atoms_with_li, 
                                    format="lammps-data", 
                                    masses=True)
                elements_list = sorted(set(atoms_with_li.get_chemical_symbols()))
                elements = " ".join(elements_list)
                Li_type = elements_list.index("Li")+1
                try:
                    shutil.copy(lammps_in_script, f"{md_dir}/in.diffusion")
                    with open(f"{md_dir}/in.diffusion", "r") as f:
                        lines = f.readlines()
                    temp_line = False
                    for i, line in enumerate(lines):
                        if re.match(r"^\s*variable\s+temp\s+equal\s+", line):
                            lines[i] = re.sub(r"(variable\s+temp\s+equal\s+).*", 
                            rf"\1{temperature}", line)
                            temp_line = True
                            break
                    with open(f"{md_dir}/in.diffusion", "w") as f:
                        f.writelines(lines)
                    assert temp_line, ""
                    logging.warning("Temperature variable line not found in LAMMPS input script. "
                                    f"Add 'variable temp equal' to {lammps_in_script}. Falling back to the default input script.")
                except (FileNotFoundError, TypeError, AssertionError):
                    with open(f"{md_dir}/in.diffusion", "w") as f:
                        f.write("units           metal\n")
                        f.write("boundary        p p p\n")
                        f.write("atom_style      atomic\n")
                        f.write("atom_modify     map yes\n")
                        f.write("newton          on\n")
                        f.write("\n")
                        f.write(f'variable        elements string "{elements}"\n')
                        f.write(f"variable        temp equal {temperature}\n")
                        f.write("\n")
                        f.write("read_data       data.structure\n")
                        f.write("\n")
                        f.write("pair_style      mace no_domain_decomposition\n")
                        f.write("pair_coeff      * * " + model + " ${elements}\n")
                        f.write("\n")
                        f.write(f"timestep        {timestep}\n")
                        f.write("\n")
                        f.write("thermo          100\n")
                        f.write("thermo_style    custom step temp pe etotal vol press fmax\n")
                        f.write("\n")
                        f.write(f"velocity        all create ${{temp}} {vrand} rot yes dist gaussian\n")
                        f.write("\n")
                        f.write("dump            equil all custom 100 equil.lammpstrj id element x y z\n")
                        f.write("dump_modify     equil append yes element ${elements}\n")
                        f.write("\n")
                        f.write("fix             1 all nvt temp ${temp} ${temp} 0.1\n")
                        f.write(f"run             {n_steps_equil}\n")
                        f.write("unfix           1\n")
                        f.write("undump          equil\n")
                        f.write("\n")
                        f.write("reset_timestep  0\n")
                        f.write("\n")
                        f.write(f"group           li type {Li_type}\n")
                        f.write("compute         msdli li msd\n")
                        f.write("compute         msdcomli li msd com yes\n")
                        f.write("\n")
                        f.write("thermo          100\n")
                        f.write("thermo_style    custom step temp pe etotal vol press fmax c_msdli[4] c_msdcomli[4]\n")
                        f.write("\n")
                        f.write("dump            diffuse all custom 100 diffuse.lammpstrj id element x y z xu yu zu\n")
                        f.write("dump_modify     diffuse append yes element ${elements}\n")
                        f.write("\n")
                        f.write("fix             1 all nvt temp ${temp} ${temp} 0.1\n")
                        f.write(f"run             {n_steps_diffuse}\n")
                        f.write("unfix           1\n")
                lammps_command = os.environ.get("LAMMPS_COMMAND")
                os.chdir(md_dir)
                complete = False
                if os.path.exists("log.lammps"):
                    with open("log.lammps", "r") as f:
                        f.seek(0, 2)
                        f_size = f.tell()
                        read_size = min(1000, f_size)
                        f.seek(max(0, f_size - read_size))
                        end_content = f.read()
                        if "Total wall time:" in end_content:
                            complete = True
                            logging.info(f"log.lammps already exists. Skipping MD {md_dir.split('/')[-2:]}...")
                if not complete:
                    subprocess.run(lammps_command, shell=True)
                os.chdir(ratio_dir)
        os.chdir(self.root_dir)
    
    def run_diffusion_calc(self, **kwargs):
        for ratio in self.li_m_ratios:
            self.setup_md(li_m_ratio=ratio, **kwargs)
    
    def run_dft_calcs(self, n_addnl_bands: int=20, n_lowest: int=3):
        self.create_dirs()
        self.run_energy_calc()
        self.run_electronic_calc(n_addnl_bands=n_addnl_bands)
        self.run_bader_calc()
        self.run_intercalation_calc(n_lowest=n_lowest)
    
    def run(self, n_addnl_bands: int=20, n_lowest: int=3, **kwargs):
        self.run_mlip_calcs()
        self.run_dft_calcs(n_addnl_bands=n_addnl_bands, n_lowest=n_lowest)
        self.run_diffusion_calc(**kwargs)
