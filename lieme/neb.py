import os
import copy
import logging
import subprocess
from typing import List, Optional
import numpy as np
import ase
from ase import Atoms
from ase.io import read, write, Trajectory
from ase.calculators.calculator import Calculator
from ase.calculators.vasp import Vasp
from ase.mep import NEB as ASENEB
from ase.optimize import BFGS
from scipy.spatial import Voronoi, _qhull
from pymatgen.io.ase import AseAtomsAdaptor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class NEBGenerator:
    def __init__(self,
                 material: str,
                 atoms: Atoms,
                 li_index: int,
                 voronoi_site_index: int = 0,
                 mlip_calc: Optional[Calculator]=None,
                 base_vasp_calc: Optional[Vasp]=None,
                 fmax: float=0.05,
                 n_images: int=6,
                 li_atom_cutoff: float=1.7,
                 ):
        """Generates and runs a Li migration NEB between a Li site and a neighboring Voronoi
            interstitial site, using either VASP or an MLIP ASE Calculator. The Voronoi interstitial
            sites are found using the same algorithm as `Intercalation.generate_intercalated_structures`
            in lieme.featurize.

        Args:
            material (str): Name of the material. This will be the name of the calculation directory.
            atoms (Atoms): ASE Atoms object of the material, containing the Li atom to be migrated.
            li_index (int): Index of the Li atom in `atoms` to shift to a neighboring Voronoi site.
            voronoi_site_index (int, optional): Index of the neighboring Voronoi interstitial site to
                shift the Li atom to, ordered by increasing distance from the Li atom (0 is the nearest
                valid site). Use this to select between multiple adjacent Voronoi sites. Defaults to 0.
            mlip_calc (Calculator, optional): MLIP ASE Calculator to use for `run_neb_mlip`. Defaults to None.
            base_vasp_calc (Vasp, optional): A configured ASE Vasp Calculator used by `run_neb_vasp` to 
                build the per-image VASP settings. NEB-specific INCAR tags are set internally on top and 
                do not need to be included. Defaults to None.
            fmax (float, optional): Force convergence criterion for the NEB. Defaults to 0.05 eV/Å.
            n_images (int, optional): Number of movable images between the fixed endpoints. Defaults to 6.
            li_atom_cutoff (float, optional): Distance cutoff below which a Voronoi vertex is considered
                to coincide with an existing atom (including other Li atoms) and is discarded. Defaults to 1.7 Å.
        """
        assert atoms[li_index].symbol == "Li", f"Atom at index {li_index} is not Li."
        self.material = material
        self.root_dir = os.getcwd()
        self.material_dir = os.path.join(self.root_dir, self.material)
        self.atoms = atoms
        self.li_index = li_index
        self.voronoi_site_index = voronoi_site_index
        self.mlip_calc = mlip_calc
        self.base_vasp_calc = base_vasp_calc
        self.fmax = fmax
        self.n_images = n_images
        self.li_atom_cutoff = li_atom_cutoff
        self.neb_dir = os.path.join(self.material_dir, "NEB", f"Li{li_index}_site{voronoi_site_index}")

    def get_voronoi_sites(self, poscar_path: str="sites.poscar") -> List[np.ndarray]:
        """Finds candidate Voronoi interstitial fractional coordinates near `self.li_index`, sorted by
            distance from that Li atom (nearest first). Sites that coincide with any existing atom
            (including other Li atoms) within `self.li_atom_cutoff` are discarded. Adapted from
            `Intercalation.generate_intercalated_structures` in lieme.featurize.

            Also writes `self.neb_dir`/poscar_path, containing `self.atoms` with a He atom placed at
            every valid Voronoi site. The He atoms are appended in the same order as the returned list,
            so the i-th He atom in the POSCAR corresponds to `valid_sites[i]`.

        Args:
            poscar_path (str, optional): File name (relative to `self.neb_dir`) to write the POSCAR to.
                Defaults to "sites.poscar".

        Returns:
            List[np.ndarray]: Fractional coordinates of valid neighboring Voronoi sites, nearest first.
        """
        structure = AseAtomsAdaptor.get_structure(self.atoms)
        lattice = structure.lattice
        image_shifts = [lattice.get_cartesian_coords([i, j, k])
                        for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
        coords = np.array([site.coords+shift for site in structure.sites for shift in image_shifts])
        try:
            voro = Voronoi(coords)
        except _qhull.QhullError:
            voro = Voronoi(coords, qhull_options="QJ")
        frac_sites = []
        for v in voro.vertices:
            f = lattice.get_fractional_coords(v)
            if np.all(f>=0) and np.all(f<=1):
                frac_sites.append(f)
        def is_valid(frac_site):
            c = structure.lattice.get_cartesian_coords(frac_site)
            dists = [np.linalg.norm(c - site.coords) for site in structure.sites]
            return all(dist>self.li_atom_cutoff for dist in dists)
        valid_sites = [f for f in frac_sites if is_valid(f)]
        li_frac = structure[self.li_index].frac_coords
        def distance_to_li(frac_site):
            dist, _ = structure.lattice.get_distance_and_image(li_frac, frac_site)
            return dist
        valid_sites = sorted(valid_sites, key=distance_to_li)
        if not valid_sites:
            raise RuntimeError(f"No valid neighboring Voronoi sites found for Li index {self.li_index} "
                               f"in {self.material}. Try lowering `li_atom_cutoff`.")
        os.makedirs(self.neb_dir, exist_ok=True)
        sites_structure = structure.copy()
        for prop in list(sites_structure.site_properties):
            sites_structure.remove_site_property(prop)
        for f in valid_sites:
            sites_structure.append("He", f, coords_are_cartesian=False)
        sites_atoms = AseAtomsAdaptor.get_atoms(sites_structure)
        write(os.path.join(self.neb_dir, poscar_path), sites_atoms, format="vasp", direct=True)
        return valid_sites

    def get_initial_atoms(self, relax: bool=False) -> Atoms:
        """Returns the initial NEB image (the Li atom at its original site).

        Args:
            relax (bool, optional): Whether to relax the atomic positions of the initial image using
                `self.mlip_calc` before returning it. Only the atomic positions are relaxed; the cell
                is kept fixed so it still matches the final image for NEB interpolation. Defaults to False.

        Returns:
            Atoms: Copy of `self.atoms`, relaxed if `relax` is True.
        """
        initial_atoms = self.atoms.copy()
        if relax:
            assert self.mlip_calc is not None, "mlip_calc must be provided to relax the initial image."
            os.makedirs(self.neb_dir, exist_ok=True)
            initial_atoms.calc = copy.deepcopy(self.mlip_calc)
            relax_traj_path = os.path.join(self.neb_dir, "relax_initial.traj")
            opt = BFGS(initial_atoms, trajectory=relax_traj_path)
            opt.run(fmax=self.fmax)
        return initial_atoms

    def get_final_atoms(self, relax: bool=False) -> Atoms:
        """Shifts the Li atom at `self.li_index` to the `self.voronoi_site_index`-th nearest neighboring
            Voronoi interstitial site.

        Args:
            relax (bool, optional): Whether to relax the atomic positions of the final image (with the
                Li atom at the target Voronoi site) using `self.mlip_calc` before returning it. Only the
                atomic positions are relaxed; the cell is kept fixed so it still matches the initial image
                for NEB interpolation. Defaults to False.

        Returns:
            Atoms: Copy of `self.atoms` with the Li atom shifted to the target Voronoi site.
        """
        voronoi_sites = self.get_voronoi_sites()
        if self.voronoi_site_index >= len(voronoi_sites):
            raise IndexError(f"voronoi_site_index={self.voronoi_site_index} out of range; only "
                             f"{len(voronoi_sites)} valid neighboring Voronoi sites found for Li index "
                             f"{self.li_index} in {self.material}.")
        target_frac = voronoi_sites[self.voronoi_site_index]
        final_atoms = self.atoms.copy()
        final_atoms.positions[self.li_index] = self.atoms.cell.cartesian_positions(target_frac)
        final_atoms.wrap()
        if relax:
            assert self.mlip_calc is not None, "mlip_calc must be provided to relax the final image."
            final_atoms.calc = copy.deepcopy(self.mlip_calc)
            relax_traj_path = os.path.join(self.neb_dir, "relax_final.traj")
            opt = BFGS(final_atoms, trajectory=relax_traj_path)
            opt.run(fmax=self.fmax)
        return final_atoms

    def get_images(self,
                   traj_path="idpp_initial.traj",
                   relax_initial: bool=False,
                   relax_final: bool=False,
                   ) -> List[Atoms]:
        """Builds the initial NEB trajectory between `self.atoms` and `self.get_final_atoms()` using
            IDPP interpolation, caching the result at `self.neb_dir`/traj_path or extracts existing
            NEB trajectory from `self.neb_dir`/traj_path.

            An existing trajectory may hold more than one optimizer step's worth of images (for example
            `neb_mlip.traj`, which `run_neb_mlip` appends a full image set to on every BFGS step), so a
            match is any length that is a whole multiple of `n_expected`, and only the last `n_expected`
            frames (the most recent/most-optimized set) are returned. This also means the file is never
            reopened in write mode once it already holds a valid trajectory, so it can't be truncated.

        Args:
            traj_path (str, optional): File name (relative to `self.neb_dir`) to read/cache the
                trajectory at. Defaults to "idpp_initial.traj".
            relax_initial (bool, optional): Whether to relax the initial image with `self.mlip_calc`
                before interpolation (see `get_initial_atoms`). Defaults to False.
            relax_final (bool, optional): Whether to relax the final image with `self.mlip_calc`
                before interpolation (see `get_final_atoms`). Defaults to False.

        Returns:
            List[Atoms]: The interpolated images, including both fixed endpoints.
        """
        os.makedirs(self.neb_dir, exist_ok=True)
        traj_path = os.path.join(self.neb_dir, traj_path)
        n_expected = self.n_images+2
        if os.path.exists(traj_path):
            try:
                traj_read = Trajectory(traj_path, "r")
                if len(traj_read)>0 and len(traj_read)%n_expected==0:
                    logging.info(f"{traj_path} already exists. Skipping IDPP interpolation...")
                    return list(traj_read)[-n_expected:]
            except ase.io.ulm.InvalidULMFileError:
                pass
        initial = self.get_initial_atoms(relax=relax_initial)
        final = self.get_final_atoms(relax=relax_final)
        images = [initial] + [initial.copy() for _ in range(self.n_images)] + [final]
        neb = ASENEB(images)
        neb.interpolate(method="idpp", mic=True)
        traj = Trajectory(traj_path, "w")
        for image in images:
            traj.write(image)
        traj.close()
        return images

    def run_neb_mlip(self,
                     climb: bool=True,
                     relax_initial: bool=False,
                     relax_final: bool=False,
                     maxstep: float=0.2,
                     ) -> List[Atoms]:
        """Runs the Li migration NEB using `self.mlip_calc`, first without and then (if `climb`) with the
            climbing image enabled.

        Args:
            climb (bool, optional): Whether to continue with climbing image NEB (CI-NEB) after the initial
                NEB converges. Defaults to True.
            relax_initial (bool, optional): Whether to relax the atomic positions of the initial image
                using `self.mlip_calc` before interpolation. Only the atomic positions are relaxed; the
                cell is kept fixed so it still matches the final image for NEB interpolation. Defaults to False.
            relax_final (bool, optional): Whether to relax the atomic positions of the final image (with the
                Li atom at the target Voronoi site) using `self.mlip_calc` before returning it. Only the
                atomic positions are relaxed; the cell is kept fixed so it still matches the initial image
                for NEB interpolation. Defaults to False.
            maxstep (float, optional): Maximum step size for the BFGS optimizer. Defaults to 0.2 Å.

        Returns:
            List[Atoms]: The relaxed NEB images, including both fixed endpoints.
        """
        assert self.mlip_calc is not None, "mlip_calc must be provided to run_neb_mlip."
        os.makedirs(self.neb_dir, exist_ok=True)
        os.chdir(self.neb_dir)
        traj_path = "neb_mlip.traj"
        n_expected = self.n_images+2
        if os.path.exists(traj_path):
            try:
                traj_read = Trajectory(traj_path, "r")
                if len(traj_read)>0 and len(traj_read)%n_expected==0:
                    logging.info(f"{traj_path} already exists. Skipping NEB at `{self.neb_dir}`...")
                    os.chdir(self.root_dir)
                    return list(traj_read)[-n_expected:]
            except ase.io.ulm.InvalidULMFileError:
                pass
        images = self.get_images(relax_initial=relax_initial, relax_final=relax_final)
        for image in images:
            image.calc = copy.deepcopy(self.mlip_calc)
        images[0].get_potential_energy()
        images[-1].get_potential_energy()
        neb = ASENEB(images)
        opt = BFGS(neb, trajectory=traj_path, maxstep=maxstep)
        opt.run(fmax=self.fmax)
        if climb:
            neb.climb = True
            opt.run(fmax=self.fmax)
        os.chdir(self.root_dir)
        return images
    
    def run_neb_vasp(self,
                     cores_per_image: int=1,
                     iopt: int=1,
                     spring: float=-5.0,
                     climb: bool=True,
                     mlip_traj: bool=False,
                     relax_final: bool=False,
                     restart: bool=False,
                     ) -> List[Atoms]:
        """Runs the Li migration NEB using VASP (VTST-style NEB implementation), writing one POSCAR
            per image into numbered subdirectories of `self.neb_dir` and a shared INCAR/KPOINTS/POTCAR
            at `self.neb_dir`. The base VASP settings come from `self.base_vasp_calc`, with NEB-specific
            INCAR tags added on top.

            With IMAGES set, VASP splits its MPI communicator into one group per movable image and
            relaxes all of them simultaneously rather than one at a time (see
            https://henkelmangroup.github.io/vtsttools/neb.html). To use this, the VASP command must be
            launched with a total number of MPI ranks that is a multiple of the number of movable images;
            `cores_per_image` sets how many ranks are dedicated to each image, and the command is built
            as `f"{VASP_MPI_LAUNCHER} -np {(len(images)-2)*cores_per_image} {VASP_EXECUTABLE}"`, reading
            `VASP_MPI_LAUNCHER` (defaults to "mpirun") and `VASP_EXECUTABLE` from the environment.

        Args:
            cores_per_image (int, optional): Number of MPI ranks dedicated to each movable image, so
                that all movable images are relaxed in parallel rather than sequentially. Defaults to 1.
            spring (float, optional): NEB spring constant (SPRING tag). Defaults to -5.0 eV/Å².
            climb (bool, optional): Whether to enable the climbing image (LCLIMB tag). Defaults to True.
            relax_final (bool, optional): Whether to relax the final image with `self.mlip_calc` before
                building the IDPP path (see `get_final_atoms`). Defaults to False.
            restart (bool, optional): If True, restart an interrupted NEB run: each movable image's
                POSCAR is written from the last ionic step of its existing OUTCAR (instead of the
                IDPP-interpolated image), and the previous run's OUTCAR and vasprun.xml for that image
                are preserved by renaming them with a numeric suffix (e.g. OUTCAR.1) before the new
                POSCAR is written. Images with no existing OUTCAR fall back to the IDPP-interpolated
                image. Defaults to False.

        Returns:
            List[Atoms]: The IDPP-interpolated images that were submitted to VASP.
        """
        assert self.base_vasp_calc is not None, "base_vasp_calc must be provided to run_neb_vasp."
        os.makedirs(self.neb_dir, exist_ok=True)
        os.chdir(self.neb_dir)
        if mlip_traj:
            images = self.get_images(traj_path="neb_mlip.traj")
        else:
            images = self.get_images(relax_final=relax_final)
        n_movable = len(images)-2
        complete = False
        first_image_outcar = os.path.join("01", "OUTCAR")
        if os.path.exists(first_image_outcar):
            with open(first_image_outcar, "r") as f:
                f.seek(0, 2)
                f_size = f.tell()
                read_size = min(2000, f_size)
                f.seek(max(0, f_size-read_size))
                end_content = f.read()
                job_finished = "General timing and accounting informations for this job:" in end_content
                converged = "reached required accuracy" in end_content
                complete = job_finished and converged
        if not complete:
            calc = copy.deepcopy(self.base_vasp_calc)
            calc.set(
                images=n_movable,
                ichain=0,
                ibrion=3,
                iopt=iopt,
                potim=0,
                isif=2,
                lclimb=climb,
                spring=spring,
                ediffg=-self.fmax,
                nsw=200,
                lcharg=False,
                lwave=False,
            )
            # `calc.write_input` below groups atoms by species in order of first
            # appearance (not alphabetically) when writing the POTCAR/POSCAR at
            # `self.neb_dir`. Each per-image POSCAR must use that exact same atom
            # ordering, or VASP silently pairs the wrong POTCAR block with each
            # image's ions (e.g. Li and Cl swapped) since it trusts ion order over
            # symbols. `calc.initialize` computes that ordering as `calc.sort`
            # without writing any files; it's the same for every image since all
            # images share the same composition.
            calc.initialize(images[1])
            vasp_sort = calc.sort
            for i, image in enumerate(images):
                image_dir = f"{i:02d}"
                os.makedirs(image_dir, exist_ok=True)
                write_image = image[vasp_sort]
                if restart and 0 < i <= n_movable:
                    outcar_path = os.path.join(image_dir, "OUTCAR")
                    if os.path.exists(outcar_path) and os.path.getsize(outcar_path)>0:
                        # Already in `vasp_sort` order, since it was written that way.
                        write_image = read(outcar_path, index=-1)
                    for fname in ("OUTCAR", "vasprun.xml"):
                        fpath = os.path.join(image_dir, fname)
                        if os.path.exists(fpath):
                            suffix = 1
                            backup_path = f"{fpath}.{suffix}"
                            while os.path.exists(backup_path):
                                suffix += 1
                                backup_path = f"{fpath}.{suffix}"
                            os.rename(fpath, backup_path)
                write(os.path.join(image_dir, "POSCAR"), write_image, format="vasp", direct=True, sort=False)
            calc.write_input(images[1])
            if os.path.exists("POSCAR"):
                os.remove("POSCAR")
            launcher = os.environ.get("VASP_MPI_LAUNCHER", "mpirun")
            vasp_executable = os.environ.get("VASP_EXECUTABLE")
            assert vasp_executable, "VASP_EXECUTABLE environment variable must be set to the VASP binary path."
            total_ranks = n_movable*cores_per_image
            vasp_command = f"{launcher} -np {total_ranks} {vasp_executable}"
            logging.info(f"Running VASP NEB with {total_ranks} total MPI ranks "
                         f"({cores_per_image} per image, {n_movable} movable images)...")
            subprocess.run(vasp_command, shell=True)
        os.chdir(self.root_dir)
        return images

    def get_vasp_neb_traj(self, traj_path: str="neb_vasp.traj") -> List[Atoms]:
        """Reads the VASP NEB images written by `run_neb_vasp` from the numbered subdirectories of
            `self.neb_dir`, attaching each movable image's energy and forces from its OUTCAR, and caches
            the result at `self.neb_dir`/traj_path.

            The two fixed endpoints (00 and the last image) are not relaxed as part of the VASP NEB chain
            and usually have no OUTCAR, so their POSCAR positions are read with no energy/forces attached,
            unless an OUTCAR happens to be present there too (for example from a separate single-point run).

        Args:
            traj_path (str, optional): File name (relative to `self.neb_dir`) to cache the result at.
                Defaults to "neb_vasp.traj".

        Returns:
            List[Atoms]: The NEB images, in order, with energy/forces attached where available.
        """
        n_expected = self.n_images+2
        images = []
        for i in range(n_expected):
            dir_path = os.path.join(self.neb_dir, f"{i:02d}")
            outcar_path = os.path.join(dir_path, "OUTCAR")
            if os.path.exists(outcar_path) and os.path.getsize(outcar_path)>0:
                with open(outcar_path, "r") as f:
                    f.seek(0, 2)
                    f_size = f.tell()
                    read_size = min(1000, f_size)
                    f.seek(max(0, f_size-read_size))
                    end_content = f.read()
                if "General timing and accounting informations for this job:" not in end_content:
                    logging.warning(f"OUTCAR at {dir_path} does not look complete/converged. "
                                    "Using its last available ionic step...")
                try:
                    image = read(outcar_path, index=-1)
                except Exception:
                    logging.warning(f"Failed to read energy/forces from OUTCAR at {dir_path}. "
                                    "Falling back to POSCAR...")
                    image = read(os.path.join(dir_path, "POSCAR"))
            else:
                image = read(os.path.join(dir_path, "POSCAR"))
            images.append(image)
        traj_path = os.path.join(self.neb_dir, traj_path)
        traj = Trajectory(traj_path, "w")
        for image in images:
            traj.write(image)
        traj.close()
        return images
